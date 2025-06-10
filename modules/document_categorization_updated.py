import streamlit as st
import logging
import json
import requests
import re
import os
import datetime
import pandas as pd
import altair as alt
from typing import Dict, Any, List, Optional, Tuple

# Import the sequential consensus implementation
from modules.sequential_consensus_implementation import categorize_document_with_sequential_consensus
from modules.document_categorization_utils import (
    categorize_document,
    extract_document_features,
    calculate_multi_factor_confidence,
    apply_confidence_calibration,
    combine_categorization_results,
    categorize_document_detailed
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UPDATED_MODEL_LIST = [
    "azure__openai__gpt_4_1_mini", "azure__openai__gpt_4_1", "azure__openai__gpt_4o_mini", "azure__openai__gpt_4o",
    "google__gemini_2_5_pro_preview", "google__gemini_2_5_flash_preview", "google__gemini_2_0_flash_001", "google__gemini_2_0_flash_lite_preview",
    "aws__claude_3_haiku", "aws__claude_3_sonnet", "aws__claude_3_5_sonnet", "aws__claude_3_7_sonnet",
    "aws__claude_4_sonnet", "aws__claude_4_opus", "aws__titan_text_lite",
    "ibm__llama_3_2_90b_vision_instruct", "ibm__llama_4_scout",
    "xai__grok_3_beta", "xai__grok_3_mini_reasoning_beta", "azure__openai__gpt_o3"
]

def document_categorization():
    """
    Main function for document categorization tab.
    """
    st.title("Document Categorization")
    
    # Initialize session state for document categorization
    if "document_categorization" not in st.session_state:
        st.session_state.document_categorization = {
            "is_categorized": False,
            "results": [],
            "errors": []
        }
    
    # Initialize document types if not already in session state
    if "document_types" not in st.session_state:
        logger.warning("Initializing/Resetting document_types in session state to default structure.")
        st.session_state.document_types = [
            {"name": "Sales Contract", "description": "Contracts related to sales agreements and terms."},
            {"name": "Invoice", "description": "Billing documents issued by a seller to a buyer, indicating quantities, prices for products or services."},
            {"name": "Tax", "description": "Documents related to government taxation (e.g., tax forms, filings, receipts)."},
            {"name": "Financial Report", "description": "Reports detailing the financial status or performance of an entity."},
            {"name": "Employment Contract", "description": "Agreements outlining terms and conditions of employment."},
            {"name": "PII", "description": "Documents containing Personally Identifiable Information that needs careful handling."},
            {"name": "Other", "description": "Any document not fitting into the specific categories above."}
        ]
    
    # Create tabs for categorization and settings
    tab1, tab2 = st.tabs(["Categorization", "Settings"])
    
    with tab1: # Categorization Tab
        st.write("## AI Model Selection")
        
        # Consensus mode selection
        st.write("### Consensus Mode")
        consensus_mode = st.radio(
            "Consensus Mode",
            ["Standard", "Parallel Consensus", "Sequential Consensus"],
            help="Standard: Single model categorization. Parallel: Multiple models categorize independently. Sequential: Models review each other's work."
        )
        
        if consensus_mode == "Standard":
            # Single model selection
            default_standard_model = "google__gemini_2_0_flash_001"
            if default_standard_model not in UPDATED_MODEL_LIST:
                default_standard_model = UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None

            model = st.selectbox(
                "Select AI Model",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_standard_model) if default_standard_model in UPDATED_MODEL_LIST else 0,
                help="Select the AI model to use for document categorization."
            )
            
            # Two-stage categorization option
            use_two_stage = st.checkbox(
                "Use two-stage categorization",
                help="First categorize as PII/non-PII, then apply specific categories to non-PII documents."
            )
            
            if use_two_stage:
                confidence_threshold = st.slider(
                    "Confidence threshold for second-stage",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.01,
                    help="Documents with first-stage confidence below this threshold will undergo second-stage categorization."
                )
            else:
                confidence_threshold = 0.6  # Default value
        
        elif consensus_mode == "Parallel Consensus":
            # Multiple model selection
            st.write("Select models for parallel consensus:")
            
            default_parallel_models = [
                m for m in ["google__gemini_2_0_flash_001", "aws__claude_3_sonnet", "azure__openai__gpt_4_1"]
                if m in UPDATED_MODEL_LIST
            ]
            if not default_parallel_models and UPDATED_MODEL_LIST: # Ensure at least one default if possible
                default_parallel_models = [UPDATED_MODEL_LIST[0]]

            models = st.multiselect(
                "Select models for parallel consensus:",
                options=UPDATED_MODEL_LIST,
                default=default_parallel_models
            )
            
            # Two-stage categorization option
            use_two_stage = st.checkbox(
                "Use two-stage categorization",
                help="First categorize as PII/non-PII, then apply specific categories to non-PII documents."
            )
            
            if use_two_stage:
                confidence_threshold = st.slider(
                    "Confidence threshold for second-stage",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.01,
                    help="Documents with first-stage confidence below this threshold will undergo second-stage categorization."
                )
            else:
                confidence_threshold = 0.6  # Default value
        
        else:  # Sequential Consensus
            st.write("### Select models for sequential consensus:")
            
            # Model 1 (Initial Analysis)
            st.write("#### Model 1 (Initial Analysis)")
            default_model1 = "google__gemini_2_0_flash_001"
            if default_model1 not in UPDATED_MODEL_LIST:
                default_model1 = UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None
            model1 = st.selectbox(
                "Model 1 (Initial Analysis)",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_model1) if default_model1 in UPDATED_MODEL_LIST else 0,
                help="This model will perform the initial document categorization."
            )
            
            # Model 2 (Expert Review)
            st.write("#### Model 2 (Expert Review)")
            default_model2 = "aws__claude_3_sonnet"
            if default_model2 not in UPDATED_MODEL_LIST:
                default_model2 = UPDATED_MODEL_LIST[1] if len(UPDATED_MODEL_LIST) > 1 else (UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None)
            model2 = st.selectbox(
                "Model 2 (Expert Review)",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_model2) if default_model2 in UPDATED_MODEL_LIST else 0,
                help="This model will review Model 1's categorization."
            )
            
            # Model 3 will be used for arbitration only when needed
            st.write("#### Model 3 will be used for arbitration only when needed:")
            
            # Model 3 (Arbitration)
            default_model3 = "aws__claude_3_5_sonnet" # Changed from anthropic__claude_3_5_sonnet
            if default_model3 not in UPDATED_MODEL_LIST:
                default_model3 = UPDATED_MODEL_LIST[2] if len(UPDATED_MODEL_LIST) > 2 else (UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None)
            model3 = st.selectbox(
                "Model 3 (Arbitration)",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_model3) if default_model3 in UPDATED_MODEL_LIST else 0,
                help="This model will arbitrate if there's significant disagreement between Models 1 and 2."
            )
            
            # Sequential Consensus Parameters
            st.write("### Sequential Consensus Parameters")
            
            # Disagreement threshold
            disagreement_threshold = st.slider(
                "Disagreement Threshold",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.01,
                help="Confidence difference threshold that triggers Model 3 arbitration."
            )
            
            # Two-stage categorization option
            use_two_stage = st.checkbox(
                "Use two-stage categorization",
                help="First categorize as PII/non-PII, then apply specific categories to non-PII documents."
            )
            
            if use_two_stage:
                confidence_threshold = st.slider(
                    "Confidence threshold for second-stage",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.01,
                    help="Documents with first-stage confidence below this threshold will undergo second-stage categorization."
                )
            else:
                confidence_threshold = 0.6  # Default value
        
        st.write("## Categorization Options")
        
        # Selection mode: Files or Folder
        selection_mode = st.radio(
            "Select files from:",
            ["Selected Files", "Box Folder"],
            help="Process files you've already selected or specify a Box folder to process all files within it."
        )
        
        if selection_mode == "Selected Files":
            # Display selected files
            if not st.session_state.selected_files:
                st.warning("No files selected. Please go to the 'Select Files' step first or choose 'Box Folder' option.")
            else:
                st.write(f"Processing {len(st.session_state.selected_files)} selected files:")
                for file in st.session_state.selected_files:
                    st.write(f"- {file['name']}")
        else:  # Box Folder
            # Folder selection
            default_folder_id_from_browser = st.session_state.get('current_folder_id', '0')
            initial_folder_id_value = default_folder_id_from_browser if default_folder_id_from_browser != '0' else "0" # Or use a specific previous default if '0' is not desired when at root. For now, "0" is fine.
            folder_id = st.text_input("Box Folder ID", value=initial_folder_id_value)
        
        # Start and cancel buttons
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("Start Categorization")
        with col2:
            cancel_button = st.button("Cancel Categorization")
        
        # Process categorization
        if start_button:
            st.session_state.document_categorization["is_categorized"] = False
            st.session_state.document_categorization["results"] = []
            st.session_state.document_categorization["errors"] = []
            
            # Get files to process based on selection mode
            files_to_process = []
            
            if selection_mode == "Selected Files":
                if not st.session_state.selected_files:
                    st.error("No files selected. Please go to the 'Select Files' step first or choose 'Box Folder' option.")
                else:
                    # Use the selected files directly
                    files_to_process = [
                        {"id": file["id"], "name": file["name"], "type": file["type"]} 
                        for file in st.session_state.selected_files
                    ]
            else:  # Box Folder
                try:
                    # Get folder contents
                    folder = st.session_state.client.folder(folder_id).get()
                    items = folder.get_items()
                    
                    # Filter for files only
                    files_to_process = [
                        {"id": item.id, "name": item.name, "type": item.type} 
                        for item in items if item.type == "file"
                    ]

                    if not files_to_process:
                        st.warning("No files found in the specified folder.")
                except Exception as e:
                    st.error(f"Error accessing folder: {str(e)}")
            
            if files_to_process:
                # Process each file
                progress_text = st.empty()
                progress_text.info(f"Processing {len(files_to_process)} files...")

                if consensus_mode == "Standard":
                    progress_text.info(f"Processing {len(files_to_process)} files with {model}...")

                    for file in files_to_process:
                        try:
                            progress_text.info(f"Processing {file['name']}...")
                            
                            if use_two_stage:
                                result = categorize_document_detailed(
                                    file["id"],
                                    model, 
                                    st.session_state.document_types,
                                    confidence_threshold
                                )
                            else:
                                result = categorize_document(
                                    file["id"],
                                    model, 
                                    st.session_state.document_types
                                )
                            
                            # Add file info to result
                            result["file_id"] = file["id"]
                            result["file_name"] = file["name"]
                            
                            # Calculate multi-factor confidence
                            document_features = extract_document_features(file["id"])
                            multi_factor_confidence = calculate_multi_factor_confidence(
                                result["confidence"],
                                document_features,
                                result["document_type"],
                                result.get("reasoning", ""),
                                [dtype["name"] for dtype in st.session_state.document_types]
                            )
                            result["multi_factor_confidence"] = multi_factor_confidence
                            
                            # Apply confidence calibration
                            calibrated_confidence = apply_confidence_calibration(
                                result["document_type"],
                                multi_factor_confidence.get("overall", result["confidence"])
                            )
                            result["calibrated_confidence"] = calibrated_confidence
                            
                            # Add to results
                            st.session_state.document_categorization["results"].append(result)
                            
                        except Exception as e:
                            logger.error(f"Error categorizing document {file['name']}: {str(e)}")
                            st.session_state.document_categorization["errors"].append({
                                "file_id": file["id"],
                                "file_name": file["name"],
                                "error": str(e)
                            })
                
                elif consensus_mode == "Parallel Consensus":
                    if not models:
                        st.error("Please select at least one model for parallel consensus.")
                        return

                    progress_text.info(f"Processing {len(files_to_process)} files with {len(models)} models in parallel...")

                    for file in files_to_process:
                        try:
                            progress_text.info(f"Processing {file['name']} with parallel consensus...")
                            
                            # Get results from all selected models
                            model_results = []
                            for model_name in models:
                                try:
                                    if use_two_stage:
                                        model_result = categorize_document_detailed(
                                            file["id"],
                                            model_name,
                                            st.session_state.document_types,
                                            confidence_threshold
                                        )
                                    else:
                                        model_result = categorize_document(
                                            file["id"],
                                            model_name,
                                            st.session_state.document_types
                                        )

                                    model_result["model_name"] = model_name
                                    model_results.append(model_result)
                                except Exception as e:
                                    logger.error(f"Error with model {model_name} for {file['name']}: {str(e)}")

                            if model_results:
                                # Combine results from all models
                                combined_result = combine_categorization_results(model_results)
                                
                                # Add file info to result
                                combined_result["file_id"] = file["id"]
                                combined_result["file_name"] = file["name"]
                                combined_result["model_results"] = model_results
                                
                                # Calculate multi-factor confidence
                                document_features = extract_document_features(file["id"])
                                multi_factor_confidence = calculate_multi_factor_confidence(
                                    combined_result["confidence"],
                                    document_features,
                                    combined_result["document_type"],
                                    combined_result.get("reasoning", ""),
                                    [dtype["name"] for dtype in st.session_state.document_types]
                                )
                                combined_result["multi_factor_confidence"] = multi_factor_confidence
                                
                                # Apply confidence calibration
                                calibrated_confidence = apply_confidence_calibration(
                                    combined_result["document_type"],
                                    multi_factor_confidence.get("overall", combined_result["confidence"])
                                )
                                combined_result["calibrated_confidence"] = calibrated_confidence
                                
                                # Add to results
                                st.session_state.document_categorization["results"].append(combined_result)
                            else:
                                raise Exception("All models failed to categorize the document")
                                
                        except Exception as e:
                            logger.error(f"Error categorizing document {file['name']} with parallel consensus: {str(e)}")
                            st.session_state.document_categorization["errors"].append({
                                "file_id": file["id"],
                                "file_name": file["name"],
                                "error": str(e)
                            })
                
                else:  # Sequential Consensus
                    progress_text.info(f"Processing {len(files_to_process)} files with sequential consensus...")

                    for file in files_to_process:
                        try:
                            progress_text.info(f"Processing {file['name']} with sequential consensus...")
                            
                            # Use the enhanced sequential consensus implementation
                            result = categorize_document_with_sequential_consensus(
                                file["id"],
                                model1,
                                model2,
                                model3,
                                st.session_state.document_types,
                                disagreement_threshold
                            )
                            
                            # Add file info to result
                            result["file_id"] = file["id"]
                            result["file_name"] = file["name"]
                            
                            # Calculate multi-factor confidence for the final result
                            if "model1_result" in result:
                                document_features = extract_document_features(file["id"])
                                multi_factor_confidence = calculate_multi_factor_confidence(
                                    result["confidence"],
                                    document_features,
                                    result["document_type"],
                                    result.get("reasoning", ""),
                                    [dtype["name"] for dtype in st.session_state.document_types]
                                )
                                result["multi_factor_confidence"] = multi_factor_confidence
                                
                                # Apply confidence calibration
                                calibrated_confidence = apply_confidence_calibration(
                                    result["document_type"],
                                    multi_factor_confidence.get("overall", result["confidence"])
                                )
                                result["calibrated_confidence"] = calibrated_confidence
                            
                            # Add to results
                            st.session_state.document_categorization["results"].append(result)
                            
                        except Exception as e:
                            logger.error(f"Error categorizing document {file['name']} with sequential consensus: {str(e)}")
                            st.session_state.document_categorization["errors"].append({
                                "file_id": file["id"],
                                "file_name": file["name"],
                                "error": str(e)
                            })
                
                # Update status
                st.session_state.document_categorization["is_categorized"] = True
                progress_text.success(f"Categorization complete! Processed {len(files_to_process)} files with {len(st.session_state.document_categorization['errors'])} errors.")
                
                if files_to_process: # Ensure we only update if files were actually processed
                    # Ensure files_to_process contains dictionaries with 'id', 'name', and 'type'
                    # This is already the case based on how files_to_process is constructed for folder mode:
                    # files_to_process = [
                    #     {"id": item.id, "name": item.name, "type": item.type}
                    #     for item in items if item.type == "file"
                    # ]
                    # And for selected files mode, it's also structured similarly:
                    # files_to_process = [
                    #     {"id": file["id"], "name": file["name"], "type": file["type"]}
                    #     for file in st.session_state.selected_files
                    # ]
                    # So, files_to_process should be suitable for direct assignment if it was populated.

                    st.session_state.selected_files = files_to_process
                    logger.info(f"Updated st.session_state.selected_files with {len(files_to_process)} categorized files.")

                # Display results
                display_categorization_results()
    
    with tab2:  # Settings Tab
        st.write("## Document Categorization Settings")
        
        # Document types editor
        st.write("### Document Types")
        st.write("Edit the document types and descriptions used for categorization:")
        
        # Create a copy of document types for editing
        if "document_types_edit" not in st.session_state:
            st.session_state.document_types_edit = st.session_state.document_types.copy()
        
        # Display existing document types with edit fields
        for i, doc_type in enumerate(st.session_state.document_types_edit):
            col1, col2, col3 = st.columns([0.3, 0.5, 0.2])
            
            with col1:
                doc_type["name"] = st.text_input(
                    "Category Name",
                    value=doc_type["name"],
                    key=f"doc_type_name_{i}"
                )
            
            with col2:
                doc_type["description"] = st.text_input(
                    "Description",
                    value=doc_type["description"],
                    key=f"doc_type_desc_{i}"
                )
            
            with col3:
                if st.button("Remove", key=f"remove_doc_type_{i}"):
                    st.session_state.document_types_edit.pop(i)
                    st.rerun()
        
        # Add new document type
        st.write("### Add New Document Type")
        col1, col2, col3 = st.columns([0.3, 0.5, 0.2])
        
        with col1:
            new_type_name = st.text_input("New Category Name", key="new_doc_type_name")
        
        with col2:
            new_type_desc = st.text_input("New Description", key="new_doc_type_desc")
        
        with col3:
            if st.button("Add", key="add_doc_type"):
                if new_type_name:
                    st.session_state.document_types_edit.append({
                        "name": new_type_name,
                        "description": new_type_desc
                    })
                    st.rerun()
        
        # Save changes
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Changes", key="save_doc_types"):
                st.session_state.document_types = st.session_state.document_types_edit.copy()
                st.success("Document types updated successfully!")
        
        with col2:
            if st.button("Reset to Default", key="reset_doc_types"):
                st.session_state.document_types = [
                    {"name": "Sales Contract", "description": "Contracts related to sales agreements and terms."},
                    {"name": "Invoice", "description": "Billing documents issued by a seller to a buyer, indicating quantities, prices for products or services."},
                    {"name": "Tax", "description": "Documents related to government taxation (e.g., tax forms, filings, receipts)."},
                    {"name": "Financial Report", "description": "Reports detailing the financial status or performance of an entity."},
                    {"name": "Employment Contract", "description": "Agreements outlining terms and conditions of employment."},
                    {"name": "PII", "description": "Documents containing Personally Identifiable Information that needs careful handling."},
                    {"name": "Other", "description": "Any document not fitting into the specific categories above."}
                ]
                st.session_state.document_types_edit = st.session_state.document_types.copy()
                st.success("Document types reset to default!")

def display_confidence_visualization(confidence_data):
    """
    Display a visual representation of confidence factors using Altair charts.
    
    Args:
        confidence_data (dict): Dictionary containing confidence factors
    """
    # Define colors for different confidence levels
    def get_color(value):
        if value >= 0.8:
            return "#28a745"  # Green
        elif value >= 0.6:
            return "#ffc107"  # Yellow/Orange
        else:
            return "#dc3545"  # Red
    
    # Prepare data for visualization
    chart_data = []
    
    # Map factor names to display names
    factor_display_names = {
        "ai_reported": "AI Model",
        "response_quality": "Response Quality",
        "category_specificity": "Category Specificity",
        "reasoning_quality": "Reasoning Quality",
        "document_features_match": "Document Features Match"
    }
    
    # Add each factor to the chart data
    for factor, value in confidence_data.items():
        if factor != "overall":
            display_name = factor_display_names.get(factor, factor.replace("_", " ").title())
            chart_data.append({
                "Factor": display_name,
                "Value": value,
                "Color": get_color(value)
            })
    
    # Sort factors by display order
    factor_order = ["AI Model", "Response Quality", "Category Specificity", "Reasoning Quality", "Document Features Match"]
    chart_data = sorted(chart_data, key=lambda x: factor_order.index(x["Factor"]) if x["Factor"] in factor_order else 999)
    
    # Create DataFrame for Altair
    df = pd.DataFrame(chart_data)
    
    # Create Altair chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Value:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Factor:N', sort=None),  # Use the custom sort order
        color=alt.Color('Color:N', scale=None)
    ).properties(
        height=30 * len(chart_data)
    )
    
    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Display the values next to the chart
    for item in chart_data:
        st.write(f"{item['Factor']}: {item['Value']:.2f}")
    
    # Display explanations for each factor
    st.write("### Confidence Factors Explained:")
    st.markdown("""
    * **AI Model**: Confidence reported directly by the AI model
    * **Response Quality**: How well-structured the AI response was
    * **Category Specificity**: How specific and definitive the category assignment is
    * **Reasoning Quality**: How detailed and specific the reasoning is
    * **Document Features Match**: How well document features match the assigned category
    """)

def display_categorization_results():
    """
    Display the results of document categorization.
    """
    if not st.session_state.document_categorization["is_categorized"]:
        return
    
    st.write("## Categorization Results")
    
    # Create tabs for table view and detailed view
    tab1, tab2 = st.tabs(["Table View", "Detailed View"])
    
    with tab1:  # Table View
        if st.session_state.document_categorization["results"]:
            # Create a DataFrame for display
            results_data = []
            
            for result in st.session_state.document_categorization["results"]:
                # Get confidence level
                confidence = result.get("calibrated_confidence", result.get("confidence", 0))
                
                # Determine confidence level text
                if confidence >= 0.8:
                    confidence_level = "High"
                elif confidence >= 0.6:
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"
                
                # Determine status based on confidence
                if confidence >= 0.85:
                    status = "Auto-Accepted"
                elif confidence >= 0.6:
                    status = "Needs Review"
                else:
                    status = "Low Confidence"
                
                # Get consensus info if available
                consensus_info = ""
                if "sequential_consensus" in result:
                    agreement_level = result["sequential_consensus"].get("agreement_level", "")
                    consensus_info = f"Sequential: {agreement_level}"
                elif "consensus_info" in result:
                    consensus_info = f"Parallel: {result['consensus_info'].get('agreement_level', '')}"
                
                # Add to results data
                results_data.append({
                    "File Name": result["file_name"],
                    "Document Type": result["document_type"],
                    "Confidence": f"{confidence:.2f}",
                    "Confidence Level": confidence_level,
                    "Status": status,
                    "Consensus": consensus_info
                })
            
            # Create DataFrame and display
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Export options
            st.download_button(
                "Export Results as CSV",
                results_df.to_csv(index=False).encode('utf-8'),
                "document_categorization_results.csv",
                "text/csv",
                key="export_results_csv"
            )
        else:
            st.info("No categorization results available.")
    
    with tab2:  # Detailed View
        if st.session_state.document_categorization["results"]:
            for result in st.session_state.document_categorization["results"]:
                with st.expander(f"{result['file_name']} - {result['document_type']}", expanded=False):
                    # Basic info
                    st.write(f"**Document Type:** {result['document_type']}")
                    
                    # Get confidence
                    confidence = result.get("calibrated_confidence", result.get("confidence", 0))
                    confidence_color = "green" if confidence >= 0.8 else ("orange" if confidence >= 0.6 else "red")
                    confidence_text = "High" if confidence >= 0.8 else ("Medium" if confidence >= 0.6 else "Low")
                    
                    st.write(f"**Overall Confidence:** <span style='color:{confidence_color}'>{confidence_text} ({confidence:.2f})</span>", unsafe_allow_html=True)
                    
                    # Display confidence breakdown if available - FIXED: No nested expanders
                    if "multi_factor_confidence" in result:
                        st.write("### Confidence Breakdown")
                        display_confidence_visualization(result["multi_factor_confidence"])
                    
                    # Sequential Consensus Details
                    if "sequential_consensus" in result:
                        st.write("### Sequential Consensus Details")
                        
                        agreement_level = result["sequential_consensus"].get("agreement_level", "Unknown")
                        st.write(f"**Agreement Level:** {agreement_level}")
                        
                        # Create tabs for each model
                        model_tabs = st.tabs(["Model 1 (Initial)", "Model 2 (Review)", "Model 3 (Arbitration)"])
                        
                        # Model 1 tab
                        with model_tabs[0]:
                            if "model1_result" in result:
                                model1 = result["model1_result"]
                                st.write(f"**Model:** {model1.get('model_name', 'Unknown')}")
                                st.write(f"**Category:** {model1.get('document_type', 'Unknown')}")
                                st.write(f"**Confidence:** {model1.get('confidence', 0):.2f}")
                                st.write("**Reasoning:**")
                                st.write(model1.get("reasoning", "No reasoning provided"))
                            else:
                                st.write("No Model 1 results available")
                        
                        # Model 2 tab
                        with model_tabs[1]:
                            if "model2_result" in result:
                                model2 = result["model2_result"]
                                st.write(f"**Model:** {model2.get('model_name', 'Unknown')}")
                                
                                # Show independent assessment if available
                                if "independent_assessment" in model2:
                                    independent = model2["independent_assessment"]
                                    st.write("#### Independent Assessment (before seeing Model 1's results)")
                                    st.write(f"**Category:** {independent.get('document_type', 'Unknown')}")
                                    st.write(f"**Confidence:** {independent.get('confidence', 0):.2f}")
                                    st.write("**Independent Reasoning:**")
                                    st.write(independent.get("reasoning", "No reasoning provided"))
                                    
                                    st.write("#### Review Assessment (after seeing Model 1's results)")
                                
                                st.write(f"**Category:** {model2.get('document_type', 'Unknown')}")
                                st.write(f"**Confidence:** {model2.get('confidence', 0):.2f}")
                                
                                if "review_assessment" in model2:
                                    review = model2["review_assessment"]
                                    st.write(f"**Agreement Level:** {review.get('agreement_level', 'Unknown')}")
                                    st.write("**Assessment:**")
                                    st.write(review.get("assessment_reasoning", "No assessment provided"))
                                
                                st.write("**Reasoning:**")
                                st.write(model2.get("reasoning", "No reasoning provided"))
                                
                                # Show confidence adjustment factors if available
                                if "confidence_adjustment_factors" in model2:
                                    st.write("#### Confidence Adjustment Factors:")
                                    factors = model2["confidence_adjustment_factors"]
                                    for factor, value in factors.items():
                                        st.write(f"- {factor.replace('_', ' ').title()}: {value:.2f}")
                            else:
                                st.write("No Model 2 results available")
                        
                        # Model 3 tab
                        with model_tabs[2]:
                            if "model3_result" in result:
                                model3 = result["model3_result"]
                                st.write(f"**Model:** {model3.get('model_name', 'Unknown')}")
                                st.write(f"**Category:** {model3.get('document_type', 'Unknown')}")
                                st.write(f"**Confidence:** {model3.get('confidence', 0):.2f}")
                                
                                if "arbitration_assessment" in model3:
                                    arb = model3["arbitration_assessment"]
                                    st.write("#### Arbitration Assessment:")
                                    st.write("**Model 1 Assessment:**")
                                    st.write(arb.get("model1_assessment", "No assessment provided"))
                                    st.write("**Model 2 Assessment:**")
                                    st.write(arb.get("model2_assessment", "No assessment provided"))
                                    st.write("**Arbitration Decision:**")
                                    st.write(arb.get("arbitration_reasoning", "No reasoning provided"))
                                
                                st.write("**Reasoning:**")
                                st.write(model3.get("reasoning", "No reasoning provided"))
                            else:
                                st.write("No arbitration was needed")
                    
                    # Parallel Consensus Details
                    elif "model_results" in result:
                        st.write("### Parallel Consensus Details")
                        
                        if "consensus_info" in result:
                            consensus = result["consensus_info"]
                            st.write(f"**Agreement Level:** {consensus.get('agreement_level', 'Unknown')}")
                            st.write(f"**Models in Agreement:** {consensus.get('models_in_agreement', 0)}")
                            st.write(f"**Total Models:** {consensus.get('total_models', 0)}")
                        
                        # Display individual model results
                        for i, model_result in enumerate(result["model_results"]):
                            st.write(f"#### Model {i+1}: {model_result.get('model_name', 'Unknown')}")
                            st.write(f"**Category:** {model_result.get('document_type', 'Unknown')}")
                            st.write(f"**Confidence:** {model_result.get('confidence', 0):.2f}")
                            st.write("**Reasoning:**")
                            st.write(model_result.get("reasoning", "No reasoning provided"))
                    
                    # Standard categorization details
                    else:
                        st.write("### Categorization Details")
                        
                        # Display reasoning
                        st.write("#### Reasoning")
                        st.write(result.get("reasoning", "No reasoning provided"))
        else:
            st.info("No categorization results available.")
    
    # Display errors if any
    if st.session_state.document_categorization["errors"]:
        st.write("## Errors")
        for error in st.session_state.document_categorization["errors"]:
            st.error(f"Error processing {error['file_name']}: {error['error']}")
