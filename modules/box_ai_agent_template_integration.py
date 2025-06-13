import streamlit as st
import logging
import json # For utility if needed

# Import from other project modules
from modules.automated_template_mapping import suggest_templates_for_category, record_historical_selection, load_json_data, save_json_data, CONFIRMED_MAPPINGS_FILE, DEFAULT_FACTOR_WEIGHTS
from modules.processing import get_fields_for_ai_from_template # To get fields once template is chosen
from modules.agent_types import ProcessingStatus # To update document status

logger = logging.getLogger(__name__)

# --- Constants ---
AGENT_TEMPLATE_CONFIG_KEY = "agent_template_selection_config"
DEFAULT_AUTO_SELECT_THRESHOLD = 0.7 # Default confidence for auto-selection

# --- Agent Integration Logic ---

def get_agent_template_config():
    if AGENT_TEMPLATE_CONFIG_KEY not in st.session_state:
        st.session_state[AGENT_TEMPLATE_CONFIG_KEY] = {
            "auto_select_threshold": DEFAULT_AUTO_SELECT_THRESHOLD,
            "human_review_template_queue": {} # file_id -> {category, suggestions, file_name}
        }
    return st.session_state[AGENT_TEMPLATE_CONFIG_KEY]

def update_agent_template_config(key, value):
    config = get_agent_template_config()
    config[key] = value

def add_to_template_review_queue(file_id, file_name, category, suggestions):
    config = get_agent_template_config()
    config["human_review_template_queue"][file_id] = {
        "file_name": file_name,
        "category": category,
        "suggestions": suggestions,
        "timestamp": st.session_state.get("current_time", "N/A") # Assuming current_time is available
    }

def remove_from_template_review_queue(file_id):
    config = get_agent_template_config()
    if file_id in config["human_review_template_queue"]:
        del config["human_review_template_queue"][file_id]

def integrate_agent_template_selection_with_box_ai_agent(agent_instance):
    if not agent_instance:
        logger.error("Box AI Agent instance not provided for template integration.")
        return

    original_process_document_method = agent_instance.process_document

    def enhanced_process_document(file_id: str, field_definitions: list, categories: list = None, **kwargs):
        logger.info(f"ATM_INTEGRATION: Enhanced process_document called for file_id: {file_id}")

        # Initialize template_to_use and fields_for_extraction
        final_template_id_for_extraction = None
        final_fields_for_extraction = field_definitions # Default to originally passed fields
        doc_category = None
        confirmed_template_id = None # Initialize here

        # --- Stage 1: Get Categorization ---
        # Attempt to get category from agent's existing results or passed arguments
        # This logic assumes that categorization either has already run and results are in agent_instance.processing_results,
        # or it's part of the initial step of original_process_document_method if called.

        # Try to get category from a potentially pre-run categorization step (e.g., from main app flow)
        if hasattr(st.session_state, 'document_categorization') and 'results' in st.session_state.document_categorization:
            cat_results_list = st.session_state.document_categorization.get('results', [])
            # Ensure cat_results_list is indeed a list and items are dicts
            if isinstance(cat_results_list, list):
                found_cat_result = next((r for r in cat_results_list if isinstance(r, dict) and r.get('file_id') == file_id), None)
                if found_cat_result:
                    doc_category = found_cat_result.get('document_type') or found_cat_result.get('category')
                    logger.info(f"ATM_INTEGRATION: Category '{doc_category}' for file {file_id} found from main categorization results.")
            else:
                logger.warning("ATM_INTEGRATION: `st.session_state.document_categorization['results']` is not a list.")


        if not doc_category:
            # If not found in main app flow, try to get it from agent's own processing_results if available (e.g. agent ran categorization already)
            if file_id in agent_instance.processing_results and \
               agent_instance.processing_results[file_id].categorization_result and \
               agent_instance.processing_results[file_id].categorization_result.get('document_type'):
                doc_category = agent_instance.processing_results[file_id].categorization_result.get('document_type')
                logger.info(f"ATM_INTEGRATION: Category '{doc_category}' for file {file_id} found from agent's existing results.")


        if not doc_category:
            logger.warning(f"ATM_INTEGRATION: No document category found for {file_id} from pre-run categorization or agent's cache to perform template mapping. Agent will use its default flow.")
            return original_process_document_method(file_id, field_definitions, categories, **kwargs)

        # --- Stage 2: Automated Template Suggestion ---
        if doc_category:
            logger.info(f"ATM_INTEGRATION: Attempting template suggestion for category '{doc_category}' on file {file_id}")
            all_templates_list = list(st.session_state.get("metadata_templates", {}).values())
            historical_data = load_json_data("atm_historical_usage.json", {})
            expected_fields_cfg = load_json_data("atm_expected_fields_per_category.json", {})
            factor_weights = st.session_state.get("atm_factor_weights", DEFAULT_FACTOR_WEIGHTS)

            confirmed_mappings = load_json_data(CONFIRMED_MAPPINGS_FILE, {})
            confirmed_template_id = confirmed_mappings.get(doc_category)

            if confirmed_template_id:
                logger.info(f"ATM_INTEGRATION: Using user-confirmed template '{confirmed_template_id}' for category '{doc_category}'.")
                final_template_id_for_extraction = confirmed_template_id
            else:
                suggestions = suggest_templates_for_category(doc_category, all_templates_list, historical_data, expected_fields_cfg, factor_weights)

                if suggestions:
                    top_suggestion = suggestions[0]
                    config = get_agent_template_config()
                    if top_suggestion["confidence"] >= config["auto_select_threshold"]:
                        final_template_id_for_extraction = top_suggestion["template_id"]
                        logger.info(f"ATM_INTEGRATION: Auto-selected template '{final_template_id_for_extraction}' for '{doc_category}' with confidence {top_suggestion['confidence']}.")
                        current_historical_data = load_json_data("atm_historical_usage.json", {})
                        record_historical_selection(doc_category, final_template_id_for_extraction, current_historical_data)
                        save_json_data("atm_historical_usage.json", current_historical_data)
                    else:
                        logger.info(f"ATM_INTEGRATION: Template suggestion for '{doc_category}' confidence {top_suggestion['confidence']} below threshold. Flagging for human review.")
                        try:
                            file_obj = agent_instance.client.file(file_id).get(fields=['name'])
                            file_name_for_queue = file_obj.name
                        except Exception as e_file:
                            logger.error(f"ATM_INTEGRATION: Could not fetch file name for {file_id}: {e_file}")
                            file_name_for_queue = f"File ID {file_id}"
                        add_to_template_review_queue(file_id, file_name_for_queue, doc_category, suggestions)
                        final_template_id_for_extraction = None
                else:
                    logger.warning(f"ATM_INTEGRATION: No template suggestions found for category '{doc_category}'.")
                    final_template_id_for_extraction = None

        # --- Stage 3: Prepare fields for extraction based on selected template ---
        if final_template_id_for_extraction:
            try:
                if final_template_id_for_extraction.startswith('enterprise_'):
                    parts = final_template_id_for_extraction.split('_', 2)
                    scope = 'enterprise'
                    template_key = parts[2] if len(parts) >= 3 else final_template_id_for_extraction
                else:
                    scope = st.session_state.get("enterprise_scope", "enterprise")
                    template_key = final_template_id_for_extraction

                retrieved_fields = get_fields_for_ai_from_template(scope, template_key)
                if retrieved_fields:
                    final_fields_for_extraction = retrieved_fields
                    logger.info(f"ATM_INTEGRATION: Using fields from template '{template_key}' for extraction for file {file_id}.")
                else:
                    logger.warning(f"ATM_INTEGRATION: Could not retrieve fields for template '{template_key}' for file {file_id}. Using default/original fields.")
            except Exception as e:
                logger.error(f"ATM_INTEGRATION: Error getting fields for template ID '{final_template_id_for_extraction}' for file {file_id}: {e}")
        else:
            logger.info(f"ATM_INTEGRATION: No specific template selected for file {file_id}. Using default/original field definitions (count: {len(field_definitions)}).")

        # --- Stage 4: Call original processing method with final fields ---
        logger.info(f"ATM_INTEGRATION: Calling original process_document for {file_id} with effective field definitions (count: {len(final_fields_for_extraction)}).")
        result = original_process_document_method(file_id, final_fields_for_extraction, categories, **kwargs)

        # --- Post-processing updates based on template selection outcome ---
        if file_id in get_agent_template_config()["human_review_template_queue"]:
            if file_id in agent_instance.processing_results and agent_instance.processing_results[file_id].status != ProcessingStatus.ERROR:
                agent_instance.processing_results[file_id].status = ProcessingStatus.HUMAN_REVIEW_REQUIRED
                current_escalation_reason = agent_instance.processing_results[file_id].escalation_reason or ""
                if "Template selection requires human review" not in current_escalation_reason:
                    agent_instance.processing_results[file_id].escalation_reason = (current_escalation_reason + " Template selection requires human review.").strip()

        if final_template_id_for_extraction and file_id in agent_instance.processing_results:
            if agent_instance.processing_results[file_id].metadata_result is None:
                agent_instance.processing_results[file_id].metadata_result = {}

            agent_instance.processing_results[file_id].metadata_result['_template_used'] = final_template_id_for_extraction
            source = "unknown"
            if confirmed_template_id and confirmed_template_id == final_template_id_for_extraction:
                source = "confirmed_user_mapping"
            elif final_template_id_for_extraction:
                source = "automated_suggestion"
            agent_instance.processing_results[file_id].metadata_result['_template_source'] = source

        return result

    agent_instance.process_document = enhanced_process_document
    agent_instance._template_integration_complete = True
    logger.info("Box AI Agent's process_document method enhanced for automated template selection.")


# --- UI Function for Agent Tab ---
def display_agent_template_selection_ui():
    st.subheader("Agent Template Selection Review & Configuration")
    config = get_agent_template_config()

    if "agent_auto_select_thresh_slider" not in st.session_state: # Use unique key for slider state
        st.session_state.agent_auto_select_thresh_slider = config["auto_select_threshold"]

    new_threshold = st.slider(
        "Confidence Threshold for Auto-Selection",
        min_value=0.0, max_value=1.0,
        value=st.session_state.agent_auto_select_thresh_slider,
        step=0.05,
        key="agent_template_thresh_slider_actual" # Actual key for widget
    )
    if new_threshold != config["auto_select_threshold"]: # Compare with actual config value
        update_agent_template_config("auto_select_threshold", new_threshold)
        st.session_state.agent_auto_select_thresh_slider = new_threshold # Update slider state
        st.success(f"Auto-selection threshold updated to {new_threshold:.2f}")


    review_queue = config["human_review_template_queue"]
    if not review_queue:
        st.info("No documents are currently awaiting template selection review by the agent.")
        return

    st.write(f"{len(review_queue)} document(s) need template selection review:")

    file_ids_to_remove_from_ui_queue = []

    for file_id, review_item in review_queue.items():
        with st.expander(f"File: {review_item['file_name']} (ID: {file_id}) - Category: {review_item['category']}"):
            st.write("Agent's Top Template Suggestions:")
            if not review_item['suggestions']:
                st.warning("No suggestions were made for this item.")
                if st.button("Dismiss from Review (No Suggestions)", key=f"btn_dismiss_nosugg_{file_id}"):
                    file_ids_to_remove_from_ui_queue.append(file_id)
                    st.rerun()
                continue

            df_suggestions_data = []
            for sugg in review_item['suggestions'][:5]:
                df_suggestions_data.append({
                    "Template Name": sugg['template_name'],
                    "Template ID": sugg['template_id'],
                    "Confidence": sugg['confidence'],
                    "Semantic": sugg['explanation']['semantic_similarity'],
                    "Field Relevance": sugg['explanation']['field_relevance'],
                    "Historical": sugg['explanation']['historical_preference_normalized']
                })
            st.dataframe(df_suggestions_data)

            available_templates_dict = st.session_state.get("metadata_templates", {})
            template_options = {"None (Process with default/no specific template)": "atm_none_option"}
            template_options.update({tpl.get('displayName', tpl.get('id')): tpl.get('id') for tpl_id, tpl in available_templates_dict.items()})

            default_selection_key_for_selectbox = "None (Process with default/no specific template)"
            if review_item['suggestions']:
                top_sugg_id_str = str(review_item['suggestions'][0]['template_id'])
                for name, id_val in template_options.items():
                    if id_val == top_sugg_id_str:
                        default_selection_key_for_selectbox = name
                        break

            selected_template_display_name = st.selectbox(
                "Choose template to apply:",
                options=list(template_options.keys()),
                index=list(template_options.keys()).index(default_selection_key_for_selectbox),
                key=f"tpl_review_{file_id}"
            )
            chosen_template_id = template_options[selected_template_display_name]

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm & Finalize Template Choice", key=f"btn_proc_{file_id}"):
                    if chosen_template_id == "atm_none_option":
                        st.info(f"File {file_id} will be processed with default/no specific template as per manual review.")
                        if "box_ai_processing_agent" in st.session_state:
                            agent = st.session_state.box_ai_processing_agent
                            if file_id in agent.processing_results:
                                reason = agent.processing_results[file_id].escalation_reason or ""
                                agent.processing_results[file_id].escalation_reason = reason.replace("Template selection requires human review.", "Template manually reviewed: No specific template chosen.").strip()
                    else:
                        st.info(f"Confirmed template '{selected_template_display_name}' for {file_id}.")
                        historical_data = load_json_data("atm_historical_usage.json", {})
                        record_historical_selection(review_item['category'], chosen_template_id, historical_data)
                        save_json_data("atm_historical_usage.json", historical_data)

                        confirmed_mappings = load_json_data(CONFIRMED_MAPPINGS_FILE, {})
                        confirmed_mappings[review_item['category']] = chosen_template_id
                        save_json_data(CONFIRMED_MAPPINGS_FILE, confirmed_mappings)

                        st.success(f"Template '{chosen_template_id}' confirmed for category '{review_item['category']}' and historical use updated.")
                        logger.info(f"Human review confirmed template {chosen_template_id} for file {file_id}.")
                        if "box_ai_processing_agent" in st.session_state:
                            agent = st.session_state.box_ai_processing_agent
                            if file_id in agent.processing_results:
                                if agent.processing_results[file_id].metadata_result is None: agent.processing_results[file_id].metadata_result = {}
                                agent.processing_results[file_id].metadata_result['_template_used'] = chosen_template_id
                                agent.processing_results[file_id].metadata_result['_template_source'] = "human_review_override"
                                reason = agent.processing_results[file_id].escalation_reason or ""
                                agent.processing_results[file_id].escalation_reason = reason.replace("Template selection requires human review.", "Template manually reviewed and confirmed.").strip()

                    file_ids_to_remove_from_ui_queue.append(file_id)
                    st.rerun()
            with col2:
                 if st.button("Dismiss (Keep Agent's Auto-Suggestion if any)", key=f"btn_dismiss_{file_id}"):
                    st.info(f"File {file_id} dismissed from active template review.")
                    file_ids_to_remove_from_ui_queue.append(file_id)
                    st.rerun()

    if file_ids_to_remove_from_ui_queue:
        for fid_to_remove in file_ids_to_remove_from_ui_queue:
            remove_from_template_review_queue(fid_to_remove)

if __name__ == '__main__':
    logger.info("Box AI Agent Template Integration module loaded.")
