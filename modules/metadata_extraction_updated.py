"""
Metadata extraction module with enhanced confidence framework integration.

This module provides the Streamlit UI for metadata extraction with
integrated enhanced confidence framework.
"""

import streamlit as st
import logging
import pandas as pd
import altair as alt
import os
import time
import json
from typing import Dict, Any, List, Optional, Tuple

# Import metadata extraction utilities
from modules.metadata_extraction_utils import (
    extract_metadata,
    extract_metadata_with_model,
    combine_metadata_results
)

# Import enhanced confidence framework integration
from modules.metadata_extraction_integration import (
    process_metadata_with_enhanced_confidence,
    display_enhanced_metadata_confidence,
    extract_document_text_from_box,
    define_field_relationships,
    define_validation_rules
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def metadata_extraction():
    """
    Main function for metadata extraction UI.
    """
    st.title("Metadata Extraction")
    
    # Initialize session state variables if they don't exist
    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = {}
    if "field_definitions" not in st.session_state:
        st.session_state.field_definitions = []
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Configuration", "Results", "Settings"])
    
    with tab1:
        st.header("Metadata Extraction Configuration")
        
        # Selection mode: Selected Files or Box Folder
        selection_mode = st.radio(
            "Selection Mode",
            ["Selected Files", "Box Folder"],
            help="Choose whether to process selected files or all files in a Box folder"
        )
        
        files_to_process = []
        
        if selection_mode == "Selected Files":
            # Use selected files from session state
            if "selected_files" in st.session_state and st.session_state.selected_files:
                st.success(f"{len(st.session_state.selected_files)} files selected")
                files_to_process = st.session_state.selected_files
            else:
                st.warning("No files selected. Please select files in the File Browser tab.")
        else:
            # Box folder selection
            folder_id = st.text_input("Box Folder ID", value="0")
            
            if folder_id:
                try:
                    folder_items = st.session_state.client.folder(folder_id=folder_id).get_items()
                    files_to_process = [item.id for item in folder_items if item.type == "file"]
                    st.success(f"Found {len(files_to_process)} files in folder")
                except Exception as e:
                    st.error(f"Error accessing folder: {str(e)}")
        
        # Field definition selection
        st.subheader("Field Definitions")
        
        # Option to load predefined field sets
        predefined_sets = {
            "Invoice": [
                {"name": "invoice_number", "type": "string", "description": "Invoice identifier"},
                {"name": "invoice_date", "type": "date", "description": "Date invoice was issued"},
                {"name": "due_date", "type": "date", "description": "Date payment is due"},
                {"name": "vendor_name", "type": "string", "description": "Name of vendor"},
                {"name": "vendor_address", "type": "string", "description": "Address of vendor"},
                {"name": "subtotal", "type": "number", "description": "Subtotal amount before tax"},
                {"name": "tax_amount", "type": "number", "description": "Tax amount"},
                {"name": "total_amount", "type": "number", "description": "Total invoice amount"}
            ],
            "Contract": [
                {"name": "contract_id", "type": "string", "description": "Contract identifier"},
                {"name": "effective_date", "type": "date", "description": "Date contract becomes effective"},
                {"name": "expiration_date", "type": "date", "description": "Date contract expires"},
                {"name": "party_1", "type": "string", "description": "First party name"},
                {"name": "party_2", "type": "string", "description": "Second party name"},
                {"name": "contract_value", "type": "number", "description": "Total contract value"}
            ],
            "Custom": []
        }
        
        selected_set = st.selectbox(
            "Field Set",
            list(predefined_sets.keys()),
            help="Select a predefined set of fields or create a custom set"
        )
        
        if selected_set == "Custom":
            # Custom field definition
            st.session_state.field_definitions = []
            
            # Add fields dynamically
            num_fields = st.number_input("Number of Fields", min_value=1, max_value=20, value=3)
            
            for i in range(num_fields):
                col1, col2, col3 = st.columns([2, 1, 3])
                
                with col1:
                    field_name = st.text_input(f"Field {i+1} Name", key=f"field_name_{i}")
                
                with col2:
                    field_type = st.selectbox(
                        f"Field {i+1} Type",
                        ["string", "date", "number", "enum"],
                        key=f"field_type_{i}"
                    )
                
                with col3:
                    field_desc = st.text_input(f"Field {i+1} Description", key=f"field_desc_{i}")
                
                if field_name:
                    field_def = {
                        "name": field_name,
                        "type": field_type,
                        "description": field_desc
                    }
                    
                    # Add options for enum type
                    if field_type == "enum":
                        options = st.text_input(
                            f"Options for {field_name} (comma-separated)",
                            key=f"field_options_{i}"
                        )
                        if options:
                            field_def["options"] = [opt.strip() for opt in options.split(",")]
                    
                    st.session_state.field_definitions.append(field_def)
        else:
            # Use predefined field set
            st.session_state.field_definitions = predefined_sets[selected_set]
            
            # Display the selected fields
            for field in st.session_state.field_definitions:
                st.markdown(f"**{field['name']}** ({field['type']}): {field['description']}")
        
        # Extraction method selection
        extraction_method = st.radio(
            "Extraction Method",
            ["Standard", "Multi-Model"],
            help="Standard: Single model extraction. Multi-Model: Multiple models extract and results are combined."
        )
        
        # Model selection based on extraction method
        available_models = [
            "azure_openai_gpt_4o_mini",
            "azure_openai_gpt_4o",
            "aws_claude_3_sonnet",
            "aws_claude_3_haiku",
            "google_gemini_1_5_pro",
            "google_gemini_1_5_flash",
            "anthropic_claude_3_opus",
            "anthropic_claude_3_sonnet",
            "anthropic_claude_3_haiku"
        ]
        
        if extraction_method == "Standard":
            model = st.selectbox("AI Model", available_models)
            models = [model]
        else:  # Multi-Model
            st.subheader("Select Models for Multi-Model Extraction")
            model_1 = st.selectbox("Model 1", available_models, index=0)
            model_2 = st.selectbox("Model 2", available_models, index=1)
            model_3 = st.selectbox("Model 3", available_models, index=2)
            models = [model_1, model_2, model_3]
        
        # Enhanced confidence framework option
        use_enhanced_confidence = st.checkbox(
            "Use Enhanced Confidence Framework",
            value=True,
            help="Enable advanced multi-factor confidence calculation with improved explanations"
        )
        
        # Process button
        if st.button("Extract Metadata"):
            if not files_to_process:
                st.error("No files selected for processing")
            elif not st.session_state.field_definitions:
                st.error("No field definitions provided")
            else:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Reset results
                st.session_state.extraction_results = {}
                
                # Process each file
                for i, file_id in enumerate(files_to_process):
                    status_text.text(f"Processing file {i+1} of {len(files_to_process)}...")
                    
                    try:
                        # Get document text for enhanced confidence (if enabled)
                        document_text = None
                        if use_enhanced_confidence:
                            try:
                                document_text = extract_document_text_from_box(
                                    st.session_state.client,
                                    file_id
                                )
                            except Exception as e:
                                logger.warning(f"Could not extract document text: {str(e)}")
                        
                        # Define validation rules and field relationships
                        validation_rules = define_validation_rules()
                        field_relationships = define_field_relationships()
                        
                        # Process based on selected method
                        if extraction_method == "Standard":
                            # Standard extraction with single model
                            result = extract_metadata(
                                file_id,
                                models[0],
                                st.session_state.field_definitions
                            )
                            
                            # Apply enhanced confidence framework if enabled
                            if use_enhanced_confidence:
                                result = process_metadata_with_enhanced_confidence(
                                    extraction_result=result,
                                    field_definitions=st.session_state.field_definitions,
                                    document_text=document_text,
                                    validation_rules=validation_rules,
                                    field_relationships=field_relationships,
                                    use_enhanced=True
                                )
                        else:  # Multi-Model
                            # Multi-model extraction
                            results = []
                            for model in models:
                                model_result = extract_metadata_with_model(
                                    file_id,
                                    model,
                                    st.session_state.field_definitions
                                )
                                results.append(model_result)
                            
                            # Combine results
                            result = combine_metadata_results(
                                results,
                                st.session_state.field_definitions
                            )
                            
                            # Apply enhanced confidence framework if enabled
                            if use_enhanced_confidence:
                                result = process_metadata_with_enhanced_confidence(
                                    extraction_result=result,
                                    field_definitions=st.session_state.field_definitions,
                                    document_text=document_text,
                                    validation_rules=validation_rules,
                                    field_relationships=field_relationships,
                                    use_enhanced=True
                                )
                        
                        # Add file info
                        try:
                            file_info = st.session_state.client.file(file_id).get()
                            result["file_name"] = file_info.name
                        except Exception as e:
                            result["file_name"] = f"File {file_id}"
                            logger.warning(f"Could not get file name: {str(e)}")
                        
                        # Store result
                        st.session_state.extraction_results[file_id] = result
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_id}: {str(e)}")
                        st.session_state.extraction_results[file_id] = {
                            "file_name": f"File {file_id}",
                            "error": str(e)
                        }
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(files_to_process))
                
                # Complete
                status_text.text("Processing complete!")
                st.success(f"Processed {len(files_to_process)} files")
                
                # Switch to results tab
                st.experimental_set_query_params(active_tab="Results")
    
    with tab2:
        st.header("Extraction Results")
        
        if not st.session_state.extraction_results:
            st.info("No results yet. Extract metadata in the Configuration tab.")
        else:
            # Display results summary
            st.subheader("Files Processed")
            
            # Create a list of files with extraction status
            file_status = []
            for file_id, result in st.session_state.extraction_results.items():
                status = "Success"
                if "error" in result:
                    status = "Error"
                
                file_status.append({
                    "File Name": result.get("file_name", f"File {file_id}"),
                    "Status": status,
                    "File ID": file_id
                })
            
            # Display as a table
            status_df = pd.DataFrame(file_status)
            st.dataframe(status_df, use_container_width=True)
            
            # Detailed view for selected file
            st.subheader("Detailed View")
            selected_file_id = st.selectbox(
                "Select File for Detailed View",
                options=list(st.session_state.extraction_results.keys()),
                format_func=lambda x: st.session_state.extraction_results[x].get("file_name", f"File {x}")
            )
            
            if selected_file_id:
                result = st.session_state.extraction_results[selected_file_id]
                
                # Display basic information
                st.markdown(f"**File Name:** {result.get('file_name', 'Unknown')}")
                
                # Check if there was an error
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display extracted metadata
                    st.subheader("Extracted Metadata")
                    
                    # Create a table for the metadata
                    metadata_rows = []
                    
                    for field_def in st.session_state.field_definitions:
                        field_name = field_def["name"]
                        
                        # Skip if field is not in result
                        if field_name not in result:
                            continue
                        
                        # Get field value and confidence
                        field_value = result[field_name]
                        
                        # Create a row for the table
                        row = {
                            "Field": field_name.replace("_", " ").title(),
                            "Value": field_value
                        }
                        
                        metadata_rows.append(row)
                    
                    # Display as a table
                    metadata_df = pd.DataFrame(metadata_rows)
                    st.dataframe(metadata_df, use_container_width=True)
                    
                    # Display detailed view for each field
                    st.subheader("Field Details")
                    
                    for field_def in st.session_state.field_definitions:
                        field_name = field_def["name"]
                        
                        # Skip if field is not in result
                        if field_name not in result:
                            continue
                        
                        # Get field value
                        field_value = result[field_name]
                        
                        # Create an expander for each field
                        with st.expander(f"{field_name.replace('_', ' ').title()}: {field_value}"):
                            # Display confidence information
                            display_enhanced_metadata_confidence(field_name, result)
                            
                            # Display field type and description
                            st.markdown(f"**Type:** {field_def['type']}")
                            if "description" in field_def:
                                st.markdown(f"**Description:** {field_def['description']}")
                    
                    # Option to export results
                    st.subheader("Export Results")
                    
                    export_format = st.selectbox(
                        "Export Format",
                        ["JSON", "CSV"],
                        help="Select format for exporting the extraction results"
                    )
                    
                    if st.button("Export"):
                        if export_format == "JSON":
                            # Export as JSON
                            export_data = {
                                field_def["name"]: result.get(field_def["name"], "")
                                for field_def in st.session_state.field_definitions
                                if field_def["name"] in result
                            }
                            
                            # Add file information
                            export_data["file_name"] = result.get("file_name", f"File {selected_file_id}")
                            export_data["file_id"] = selected_file_id
                            
                            # Convert to JSON string
                            json_str = json.dumps(export_data, indent=2)
                            
                            # Display for download
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name=f"metadata_{selected_file_id}.json",
                                mime="application/json"
                            )
                        else:  # CSV
                            # Export as CSV
                            export_data = []
                            
                            for field_def in st.session_state.field_definitions:
                                field_name = field_def["name"]
                                if field_name in result:
                                    export_data.append({
                                        "Field": field_name,
                                        "Value": result[field_name]
                                    })
                            
                            # Convert to CSV
                            csv_df = pd.DataFrame(export_data)
                            csv_str = csv_df.to_csv(index=False)
                            
                            # Display for download
                            st.download_button(
                                label="Download CSV",
                                data=csv_str,
                                file_name=f"metadata_{selected_file_id}.csv",
                                mime="text/csv"
                            )
    
    with tab3:
        st.header("Metadata Extraction Settings")
        
        # Enhanced confidence framework settings
        st.subheader("Enhanced Confidence Framework")
        
        st.info("""
        The enhanced confidence framework for metadata extraction uses the following factors:
        
        - **AI Reported Confidence**: Raw confidence score reported by the AI model
        - **Format Validity**: Whether the extracted value matches the expected format
        - **Content Plausibility**: Whether the value is plausible for the field type
        - **Extraction Consistency**: Whether the value is consistently found in the document
        
        These factors are combined using a weighted average to produce the overall confidence score.
        
        Additionally, cross-field validation is applied to ensure consistency between related fields,
        such as date sequences and mathematical relationships.
        """)
        
        # Field relationship settings
        st.subheader("Field Relationships")
        
        st.markdown("""
        The following field relationships are used for cross-validation:
        
        1. **Date Sequence**: Due date should be after invoice date
        2. **Sum Relationship**: Subtotal + tax should approximately equal total
        3. **Distinct Values**: Invoice number and PO number should be different
        4. **Co-presence**: If vendor name exists, vendor address likely exists too
        
        These relationships are used to adjust confidence scores based on cross-field consistency.
        """)
