import streamlit as st
import json
import os
import logging
from difflib import SequenceMatcher # For a slightly better basic similarity

logger = logging.getLogger(__name__)

# --- Constants & Configuration ---
CONFIG_DIR = "config"
HISTORICAL_USAGE_FILE = os.path.join(CONFIG_DIR, "atm_historical_usage.json")
EXPECTED_FIELDS_FILE = os.path.join(CONFIG_DIR, "atm_expected_fields_per_category.json")
CONFIRMED_MAPPINGS_FILE = os.path.join(CONFIG_DIR, "atm_confirmed_mappings.json")

DEFAULT_FACTOR_WEIGHTS = {
    "semantic": 0.4,
    "field_relevance": 0.4,
    "historical": 0.2
}

# --- Data Loading/Saving ---
def _ensure_config_dir():
    if not os.path.exists(CONFIG_DIR):
        try:
            os.makedirs(CONFIG_DIR)
            logger.info(f"Created config directory: {CONFIG_DIR}")
        except OSError as e:
            logger.error(f"Error creating config directory {CONFIG_DIR}: {e}")
            # Fallback to current dir if config creation fails, not ideal but allows continuation
            return ""
    return CONFIG_DIR

def load_json_data(filename, default_data=None):
    config_dir_path = _ensure_config_dir()
    filepath = os.path.join(config_dir_path, filename) if config_dir_path else filename

    if default_data is None:
        default_data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading {filepath}: {e}")
            return default_data
    return default_data

def save_json_data(filename, data):
    config_dir_path = _ensure_config_dir()
    filepath = os.path.join(config_dir_path, filename) if config_dir_path else filename
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logger.error(f"Error saving {filepath}: {e}")

# --- Core Mapping Logic ---
def calculate_semantic_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()

def calculate_field_relevance(category_name, template_field_keys, expected_fields_config):
    if not category_name or not template_field_keys: # check template_field_keys for empty list
        return 0.0
    expected_fields = expected_fields_config.get(category_name, [])
    if not expected_fields: # If no expected fields are defined for the category
        return 0.0 # Or 0.5 if we want to be neutral if not defined? For now, 0.

    template_field_keys_set = set(template_field_keys)
    expected_fields_set = set(expected_fields)

    matched_fields = len(template_field_keys_set.intersection(expected_fields_set))

    # Consider relevance based on how many of the *expected* fields are present
    return matched_fields / len(expected_fields_set) if expected_fields_set else 0.0


def get_historical_preference(category_name, template_id, historical_data):
    category_history = historical_data.get(category_name, {})
    return category_history.get(str(template_id), 0) # Ensure template_id is string for key matching

def suggest_templates_for_category(category_name, all_templates_list, historical_data, expected_fields_config, factor_weights):
    if not all_templates_list:
        return []
    suggestions = []

    max_historical_score_for_category = 0
    category_specific_history = historical_data.get(category_name, {})
    if category_specific_history:
        # Ensure values are numbers before max()
        valid_scores = [s for s in category_specific_history.values() if isinstance(s, (int, float))]
        if valid_scores:
            max_historical_score_for_category = max(valid_scores)


    for template in all_templates_list:
        template_id = template.get('id') or template.get('template_key') # Handle both possible ID keys
        if not template_id:
            logger.warning(f"Skipping template without ID or template_key: {template.get('displayName', 'Unnamed Template')}")
            continue

        template_name = template.get('displayName', template.get('template_key', 'Unknown Template'))
        template_field_keys = [f.get('key') for f in template.get('fields', []) if f.get('key')]

        semantic_score = calculate_semantic_similarity(category_name, template_name)
        field_score = calculate_field_relevance(category_name, template_field_keys, expected_fields_config)

        historical_raw_score = get_historical_preference(category_name, str(template_id), historical_data)
        normalized_historical_score = (historical_raw_score / max_historical_score_for_category) if max_historical_score_for_category > 0 else 0.0

        # Ensure weights sum to 1 (or normalize if they don't, for simplicity assume they do for now)
        # total_weight = sum(factor_weights.values()) # Could be used for normalization

        overall_confidence = (semantic_score * factor_weights.get("semantic", 0.0) +
                              field_score * factor_weights.get("field_relevance", 0.0) +
                              normalized_historical_score * factor_weights.get("historical", 0.0))

        suggestions.append({
            "template_id": str(template_id), # Store as string
            "template_name": template_name,
            "confidence": round(overall_confidence, 3),
            "explanation": {
                "semantic_similarity": round(semantic_score, 3),
                "field_relevance": round(field_score, 3),
                "historical_preference_raw": historical_raw_score,
                "historical_preference_normalized": round(normalized_historical_score, 3),
                "weights_used": factor_weights
            }
        })
    return sorted(suggestions, key=lambda x: x["confidence"], reverse=True)

def record_historical_selection(category_name, template_id, historical_data_ref):
    # Pass historical_data as a reference (dict) to modify it directly
    template_id_str = str(template_id)
    category_history = historical_data_ref.setdefault(category_name, {})
    category_history.setdefault(template_id_str, 0)
    category_history[template_id_str] += 1
    # Save is handled by the caller UI function after potential multiple updates

# --- UI Function ---
def display_automated_mapping_ui():
    st.header("Automated Metadata Template Mapping")

    # Initialize session state for weights if not present
    if "atm_factor_weights" not in st.session_state:
        st.session_state.atm_factor_weights = DEFAULT_FACTOR_WEIGHTS.copy()

    factor_weights = st.session_state.atm_factor_weights

    # Load data
    historical_data = load_json_data("atm_historical_usage.json", {})
    expected_fields_config = load_json_data("atm_expected_fields_per_category.json", {"Example Category": ["field1", "field2"]})
    confirmed_mappings = load_json_data("atm_confirmed_mappings.json", {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Factor Weights")
        factor_weights["semantic"] = st.slider("Semantic Similarity", 0.0, 1.0, factor_weights["semantic"], 0.05, key="atm_semantic_w_disp")
        factor_weights["field_relevance"] = st.slider("Field Relevance", 0.0, 1.0, factor_weights["field_relevance"], 0.05, key="atm_field_w_disp")
        factor_weights["historical"] = st.slider("Historical Usage", 0.0, 1.0, factor_weights["historical"], 0.05, key="atm_hist_w_disp")

    with col2:
        st.subheader("Current Confirmed Mappings")
        if not confirmed_mappings:
            st.caption("No mappings confirmed yet.")
        else:
            # Display current confirmed mappings more nicely
            for cat, tpl_id in confirmed_mappings.items():
                tpl_name = "Unknown Template"
                # Attempt to find template name from session state if available
                if "metadata_templates" in st.session_state:
                     tpl = st.session_state.metadata_templates.get(tpl_id) # Assuming tpl_id is the key in metadata_templates
                     if tpl: tpl_name = tpl.get('displayName', tpl_id)
                st.markdown(f"- **{cat}** -> `{tpl_name}` (`{tpl_id}`)")


    st.subheader("Expected Fields per Document Category")
    with st.expander("Edit JSON for Expected Fields", expanded=False):
        edited_expected_fields_str = st.text_area("Expected Fields JSON:", value=json.dumps(expected_fields_config, indent=2), height=200, key="atm_exp_fields_json")
        if st.button("Save Expected Fields Configuration"):
            try:
                new_expected_fields = json.loads(edited_expected_fields_str)
                save_json_data("atm_expected_fields_per_category.json", new_expected_fields)
                st.success("Expected fields configuration saved!")
                expected_fields_config = new_expected_fields # update in-memory copy
                st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON format for expected fields.")

    st.subheader("Mapping Analysis & Confirmation")

    # Get document categories from session state (populated by categorization module)
    doc_categories_from_state = []
    if hasattr(st.session_state, 'document_types') and st.session_state.document_types:
         doc_categories_from_state = [dt.get("name") for dt in st.session_state.document_types if dt.get("name")]

    if not doc_categories_from_state:
        st.warning("No document categories found in `st.session_state.document_types`. Please define/run categorization first.")

    # Get all templates from session state
    all_metadata_templates_dict = st.session_state.get("metadata_templates", {})
    all_metadata_templates_list = []
    if isinstance(all_metadata_templates_dict, dict):
        all_metadata_templates_list = list(all_metadata_templates_dict.values())
    else:
        st.warning("`st.session_state.metadata_templates` is not a dictionary. Cannot retrieve templates.")

    if not all_metadata_templates_list:
        st.warning("No metadata templates loaded in `st.session_state.metadata_templates`. Please refresh templates.")

    if st.button("Run Automated Mapping Analysis", disabled=(not doc_categories_from_state or not all_metadata_templates_list)):
        analysis_results = {}
        if not doc_categories_from_state: st.error("Cannot run analysis: No document categories loaded.")
        if not all_metadata_templates_list: st.error("Cannot run analysis: No templates loaded.")

        if doc_categories_from_state and all_metadata_templates_list:
            with st.spinner("Analyzing mappings..."):
                for cat_name in doc_categories_from_state:
                    suggestions = suggest_templates_for_category(cat_name, all_metadata_templates_list, historical_data, expected_fields_config, factor_weights)
                    analysis_results[cat_name] = suggestions
            st.session_state.atm_analysis_results = analysis_results
            if not analysis_results:
                 st.info("Analysis complete, but no suggestions were generated.")
            else:
                 st.success("Analysis complete.")
        st.rerun()


    if "atm_analysis_results" in st.session_state and st.session_state.atm_analysis_results:
        results = st.session_state.atm_analysis_results
        st.markdown("---")

        for category, suggestions in results.items():
            st.markdown(f"#### Suggestions for: **{category}**")
            if suggestions:
                # Prepare options for selectbox: {DisplayName (ID): id, ...}
                # Ensure "None" option is first and handled correctly
                template_options_for_select = {"None (Do not map automatically)": "atm_none"}
                for s in suggestions:
                    # Use a combined name if display name and ID are different and informative
                    option_label = f"{s['template_name']} ({s['template_id']})"
                    template_options_for_select[option_label] = str(s['template_id'])

                # Determine current confirmed mapping to set as default in selectbox
                current_confirmed_id_for_cat = str(confirmed_mappings.get(category, "atm_none"))

                # Find the label corresponding to the current_confirmed_id_for_cat
                current_selection_label = "None (Do not map automatically)" # Default if not found
                for label, id_val in template_options_for_select.items():
                    if id_val == current_confirmed_id_for_cat:
                        current_selection_label = label
                        break

                # If the currently confirmed ID is not in suggestions, add it to options
                if current_confirmed_id_for_cat != "atm_none" and current_selection_label == "None (Do not map automatically)":
                    # Attempt to find its name
                    confirmed_tpl_name = "Previously Confirmed Template"
                    if "metadata_templates" in st.session_state:
                         tpl = st.session_state.metadata_templates.get(current_confirmed_id_for_cat)
                         if tpl: confirmed_tpl_name = tpl.get('displayName', current_confirmed_id_for_cat)

                    missing_label = f"{confirmed_tpl_name} ({current_confirmed_id_for_cat}) [Current Confirmed]"
                    template_options_for_select[missing_label] = current_confirmed_id_for_cat
                    current_selection_label = missing_label


                selected_option_label = st.selectbox(
                    f"Confirm template for **{category}** (Top suggestion: {suggestions[0]['template_name']} - {suggestions[0]['confidence']:.2f})",
                    options=list(template_options_for_select.keys()),
                    index=list(template_options_for_select.keys()).index(current_selection_label),
                    key=f"confirm_sel_{category}"
                )

                selected_template_id = template_options_for_select[selected_option_label]

                if st.button(f"Save Confirmation for {category}", key=f"btn_confirm_{category}"):
                    if selected_template_id == "atm_none":
                        if category in confirmed_mappings: # Remove if "None" is chosen
                            del confirmed_mappings[category]
                        st.info(f"Mapping for '{category}' removed (set to None).")
                    else:
                        confirmed_mappings[category] = selected_template_id
                        st.success(f"Mapping for '{category}' confirmed to '{selected_option_label}'.")
                        # Record historical selection (pass the historical_data dict to be modified)
                        record_historical_selection(category, selected_template_id, historical_data)
                        save_json_data("atm_historical_usage.json", historical_data) # Save updated history
                        st.info(f"Historical usage updated for {category} -> {selected_option_label}.")

                    save_json_data("atm_confirmed_mappings.json", confirmed_mappings)
                    st.rerun() # Rerun to reflect changes

                with st.expander("View all suggestions and details"):
                    st.dataframe(suggestions)
                    if suggestions: st.json(suggestions[0]['explanation'])

            else:
                st.write("No suggestions for this category.")
            st.markdown("---")

if __name__ == '__main__':
    # This block is for potential local testing of logic if needed.
    # Example Usage (Illustrative - requires Streamlit context and session_state setup)
    # pass
    logger.info("Automated Template Mapping module loaded.")
