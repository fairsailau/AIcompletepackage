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

        final_template_id_for_extraction = None
        final_fields_for_extraction = field_definitions # Default
        doc_category = None

        # Try to get category from agent's existing results
        if file_id in agent_instance.processing_results and agent_instance.processing_results[file_id].categorization_result:
            doc_category = agent_instance.processing_results[file_id].categorization_result.get('document_type')

        logger.info(f"ATM_INTEGRATION: File: {file_id} - Determined document category: {doc_category}")

        if not doc_category:
            logger.warning(f"ATM_INTEGRATION: File: {file_id} - No document category found. Skipping template mapping. Using default/original fields.")
            # Call original method (ensure this path is robust in the actual module)
            return original_process_document_method(file_id, final_fields_for_extraction, categories, **kwargs)

        logger.info(f"ATM_INTEGRATION: File: {file_id} - Attempting template suggestion for category '{doc_category}'.")

        # Load configurations (these will be empty {} if files can't be created by subtasks)
        confirmed_mappings = load_json_data("atm_confirmed_mappings.json", {})
        logger.info(f"ATM_INTEGRATION: File: {file_id} - Loaded confirmed_mappings: {confirmed_mappings}")

        historical_data = load_json_data("atm_historical_usage.json", {})
        logger.info(f"ATM_INTEGRATION: File: {file_id} - Loaded historical_data: {historical_data}")

        expected_fields_cfg = load_json_data("atm_expected_fields_per_category.json", {})
        logger.info(f"ATM_INTEGRATION: File: {file_id} - Loaded expected_fields_cfg: {expected_fields_cfg}")

        all_templates_list = list(st.session_state.get("metadata_templates", {}).values())
        logger.info(f"ATM_INTEGRATION: File: {file_id} - Number of all_metadata_templates_list: {len(all_templates_list)}")

        factor_weights = st.session_state.get("atm_factor_weights", DEFAULT_FACTOR_WEIGHTS) # DEFAULT_FACTOR_WEIGHTS needs to be accessible or defined
        logger.info(f"ATM_INTEGRATION: File: {file_id} - Using factor_weights: {factor_weights}")

        confirmed_template_id = confirmed_mappings.get(doc_category)
        logger.info(f"ATM_INTEGRATION: File: {file_id} - Confirmed template for '{doc_category}': {confirmed_template_id}")

        if confirmed_template_id:
            final_template_id_for_extraction = confirmed_template_id
            logger.info(f"ATM_INTEGRATION: File: {file_id} - Using user-confirmed template: '{final_template_id_for_extraction}'.")
        else:
            logger.info(f"ATM_INTEGRATION: File: {file_id} - No confirmed template. Running suggestion logic.")
            suggestions = suggest_templates_for_category(doc_category, all_templates_list, historical_data, expected_fields_cfg, factor_weights)
            logger.info(f"ATM_INTEGRATION: File: {file_id} - Suggestions for '{doc_category}': {suggestions[:3]}") # Log top 3 suggestions

            if suggestions:
                top_suggestion = suggestions[0]
                agent_config = get_agent_template_config() # Ensure this function is robust
                auto_select_threshold = agent_config.get("auto_select_threshold", DEFAULT_AUTO_SELECT_THRESHOLD)
                logger.info(f"ATM_INTEGRATION: File: {file_id} - Auto-select threshold: {auto_select_threshold}, Top suggestion confidence: {top_suggestion['confidence']}")

                if top_suggestion["confidence"] >= auto_select_threshold:
                    final_template_id_for_extraction = top_suggestion["template_id"]
                    logger.info(f"ATM_INTEGRATION: File: {file_id} - Auto-selected template '{final_template_id_for_extraction}' for '{doc_category}'.")
                    # (Recording historical selection logic here...)
                else:
                    logger.info(f"ATM_INTEGRATION: File: {file_id} - Suggestion confidence {top_suggestion['confidence']} below threshold. Flagging for human review.")
                    # (Flagging for human review logic here...)
                    final_template_id_for_extraction = None
            else:
                logger.warning(f"ATM_INTEGRATION: File: {file_id} - No template suggestions found for category '{doc_category}'.")
                final_template_id_for_extraction = None

        logger.info(f"ATM_INTEGRATION: File: {file_id} - Final template ID chosen for extraction: {final_template_id_for_extraction}")

        if final_template_id_for_extraction:
            try:
                # (Scope and template_key parsing logic...)
                # For logging, let's assume scope and template_key are parsed correctly
                scope = "enterprise" # Placeholder
                template_key = final_template_id_for_extraction # Placeholder if not prefixed
                if final_template_id_for_extraction.startswith('enterprise_'):
                    parts = final_template_id_for_extraction.split('_', 2)
                    template_key = parts[2] if len(parts) >= 3 else final_template_id_for_extraction

                logger.info(f"ATM_INTEGRATION: File: {file_id} - Attempting to get fields for template scope '{scope}', key '{template_key}'.")
                retrieved_fields = get_fields_for_ai_from_template(scope, template_key)

                if retrieved_fields:
                    final_fields_for_extraction = retrieved_fields
                    logger.info(f"ATM_INTEGRATION: File: {file_id} - Retrieved {len(retrieved_fields)} fields from template '{template_key}'.")
                else:
                    logger.warning(f"ATM_INTEGRATION: File: {file_id} - Could not retrieve fields for template '{template_key}'. Field list for extraction might be empty or default.")
                    final_fields_for_extraction = [] # Explicitly set to empty if no fields from template
            except Exception as e:
                logger.error(f"ATM_INTEGRATION: File: {file_id} - Error getting fields for template ID '{final_template_id_for_extraction}': {e}", exc_info=True)
                final_fields_for_extraction = [] # Explicitly set to empty on error
        else:
            logger.warning(f"ATM_INTEGRATION: File: {file_id} - No template ID selected, so no specific template fields will be fetched. Using default/original fields (count: {len(final_fields_for_extraction or [])}).")
            # If final_template_id_for_extraction is None, final_fields_for_extraction remains what was passed or its default.
            # If the intention is to extract NO fields if no template, then:
            final_fields_for_extraction = []


        metadata_field_keys_for_workflow = [] # Default to empty list
        if final_fields_for_extraction and isinstance(final_fields_for_extraction, list):
            metadata_field_keys_for_workflow = [f.get('key') for f in final_fields_for_extraction if isinstance(f, dict) and f.get('key')]

        logger.info(f"ATM_INTEGRATION: File: {file_id} - Number of metadata field keys for workflow: {len(metadata_field_keys_for_workflow)}. Keys: {metadata_field_keys_for_workflow}")

        # ... (rest of the function, including call to original_process_document_method and result updates)
        result = original_process_document_method(file_id, final_fields_for_extraction, categories, **kwargs) # Pass List[Dict]

        # ... (status updates and storing template_id_used logic ...)
        if final_template_id_for_extraction and file_id in agent_instance.processing_results:
            if agent_instance.processing_results[file_id].metadata_result is None:
                agent_instance.processing_results[file_id].metadata_result = {}
            agent_instance.processing_results[file_id].metadata_result['_template_used'] = final_template_id_for_extraction
            # Consider adding source of template selection (confirmed, auto, human_review_override)
            # This part was already good.

        return result

    agent_instance.process_document = enhanced_process_document
    logger.info("Box AI Agent's process_document method enhanced with detailed logging for template selection.")


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
