"""
Integration module for enhanced confidence framework with metadata extraction.

This module integrates the enhanced confidence framework with the existing
metadata extraction functionality, ensuring backward compatibility.
"""

import streamlit as st
import logging
import pandas as pd
import altair as alt
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union

# Import the enhanced confidence framework
from modules.enhanced_confidence_framework import (
    calculate_metadata_field_confidence,
    validate_metadata_cross_field,
    get_metadata_confidence_explanation,
    create_metadata_confidence_visualization,
    get_confidence_color_name,
    get_confidence_color_hex
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_metadata_with_enhanced_confidence(
    extraction_result: Dict[str, Any],
    field_definitions: List[Dict[str, Any]],
    document_text: Optional[str] = None,
    validation_rules: Optional[Dict[str, Any]] = None,
    field_relationships: Optional[List[Dict[str, Any]]] = None,
    use_enhanced: bool = True
) -> Dict[str, Any]:
    """
    Process metadata extraction results with enhanced confidence framework.
    
    Args:
        extraction_result: Original metadata extraction result
        field_definitions: Definitions of metadata fields
        document_text: Optional text content of the document
        validation_rules: Optional validation rules for fields
        field_relationships: Optional relationships between fields
        use_enhanced: Whether to use enhanced confidence framework
        
    Returns:
        Updated extraction result with enhanced confidence scores
    """
    if not use_enhanced:
        # Return original result for backward compatibility
        return extraction_result
    
    # Create a copy of the original result to avoid modifying it
    result = extraction_result.copy()
    
    # Create a dictionary to store field values (without _confidence suffix)
    field_values = {}
    field_confidence_scores = {}
    
    # Process each field
    for field_def in field_definitions:
        field_name = field_def.get("name", field_def.get("key", ""))
        if not field_name:
            continue
        
        field_type = field_def.get("type", "string")
        
        # Skip if field is not in extraction result
        if field_name not in result:
            continue
        
        # Get extracted value and original confidence
        extracted_value = result[field_name]
        field_values[field_name] = extracted_value
        
        # Get original confidence (High, Medium, Low)
        original_confidence = result.get(f"{field_name}_confidence", "Low")
        
        # Calculate enhanced confidence
        confidence_result = calculate_metadata_field_confidence(
            field_name=field_name,
            field_type=field_type,
            extracted_value=extracted_value,
            ai_confidence=original_confidence,
            document_text=document_text,
            field_definition=field_def,
            validation_rules=validation_rules
        )
        
        # Store confidence result
        field_confidence_scores[field_name] = confidence_result
        
        # Update result with enhanced confidence
        result[f"{field_name}_enhanced_confidence"] = confidence_result
        
        # For backward compatibility, keep original confidence
        # but also add numeric confidence
        result[f"{field_name}_confidence_numeric"] = confidence_result["numeric_confidence"]
    
    # Apply cross-field validation if relationships provided
    if field_relationships and field_confidence_scores:
        updated_confidence_scores = validate_metadata_cross_field(
            extracted_fields=field_values,
            confidence_scores=field_confidence_scores,
            field_relationships=field_relationships
        )
        
        # Update result with cross-validated confidence scores
        for field_name, confidence_result in updated_confidence_scores.items():
            result[f"{field_name}_enhanced_confidence"] = confidence_result
            result[f"{field_name}_confidence_numeric"] = confidence_result["numeric_confidence"]
            
            # For backward compatibility, update categorical confidence if it changed
            if confidence_result["categorical_confidence"] != result.get(f"{field_name}_confidence", "Low"):
                result[f"{field_name}_confidence"] = confidence_result["categorical_confidence"]
    
    # Add metadata to indicate enhanced confidence was applied
    result["_confidence_framework"] = "enhanced"
    
    return result

def display_enhanced_metadata_confidence(
    field_name: str,
    extraction_result: Dict[str, Any],
    show_explanations: bool = True
) -> None:
    """
    Display enhanced confidence visualization for a metadata field in Streamlit.
    
    Args:
        field_name: Name of the metadata field
        extraction_result: Extraction result with confidence information
        show_explanations: Whether to show explanations
    """
    # Check if enhanced confidence is available
    enhanced_confidence_key = f"{field_name}_enhanced_confidence"
    if enhanced_confidence_key in extraction_result:
        # Enhanced confidence framework
        confidence_result = extraction_result[enhanced_confidence_key]
        numeric_confidence = confidence_result.get("numeric_confidence", 0.0)
        categorical_confidence = confidence_result.get("categorical_confidence", "Low")
        
        # Display confidence indicator
        confidence_color = get_confidence_color_hex(numeric_confidence)
        confidence_label = categorical_confidence
        
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            ">
                <div style="
                    background-color: {confidence_color};
                    width: 15px;
                    height: 15px;
                    border-radius: 50%;
                    margin-right: 5px;
                "></div>
                <span style="font-weight: bold;">{confidence_label} Confidence ({numeric_confidence:.2f})</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display confidence factors if requested
        if show_explanations:
            with st.expander("Confidence Details", expanded=False):
                # Display confidence factors visualization
                chart = create_metadata_confidence_visualization(confidence_result)
                st.altair_chart(chart, use_container_width=True)
                
                # Display explanations
                st.subheader("Confidence Factor Explanations")
                explanations = get_metadata_confidence_explanation(confidence_result, field_name)
                for factor, explanation in explanations.items():
                    if factor != "overall":  # Overall is shown separately
                        st.markdown(f"**{factor.replace('_', ' ').title()}**: {explanation}")
    
    else:
        # Original confidence framework
        confidence_key = f"{field_name}_confidence"
        if confidence_key in extraction_result:
            confidence = extraction_result[confidence_key]
            
            # Map categorical confidence to color
            color_map = {
                "High": "#66BB6A",  # Green
                "Medium": "#FFA726",  # Orange
                "Low": "#FF5252"  # Red
            }
            confidence_color = color_map.get(confidence, "#9E9E9E")  # Default gray
            
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    margin-bottom: 10px;
                ">
                    <div style="
                        background-color: {confidence_color};
                        width: 15px;
                        height: 15px;
                        border-radius: 50%;
                        margin-right: 5px;
                    "></div>
                    <span style="font-weight: bold;">{confidence} Confidence</span>
                </div>
                """,
                unsafe_allow_html=True
            )

def get_field_type_from_definition(field_name: str, field_definitions: List[Dict[str, Any]]) -> str:
    """
    Get the type of a field from its definition.
    
    Args:
        field_name: Name of the field
        field_definitions: List of field definitions
        
    Returns:
        Field type (string, date, number, enum, etc.)
    """
    for field_def in field_definitions:
        if field_def.get("name", field_def.get("key", "")) == field_name:
            return field_def.get("type", "string")
    return "string"  # Default to string if not found

def extract_document_text_from_box(client: Any, file_id: str) -> str:
    """
    Extract text content from a Box document for enhanced confidence calculation.
    
    Args:
        client: Box API client
        file_id: Box file ID
        
    Returns:
        Text content of the document
    """
    try:
        # Try to get text representation from Box
        access_token = None
        if hasattr(client, '_oauth'):
            access_token = client._oauth.access_token
        elif hasattr(client, 'auth') and hasattr(client.auth, 'access_token'):
            access_token = client.auth.access_token
        
        if not access_token:
            logger.warning("Could not retrieve access token for text extraction")
            return ""
        
        headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
        
        # Use Box AI API to get text representation
        api_url = 'https://api.box.com/2.0/ai/text_representation'
        request_body = {
            'items': [{'id': file_id, 'type': 'file'}]
        }
        
        import requests
        response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
        
        if response.status_code == 200:
            response_data = response.json()
            if 'text' in response_data:
                return response_data['text']
        
        logger.warning(f"Failed to extract text from Box document: {response.status_code}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from Box document: {str(e)}")
        return ""

def define_field_relationships() -> List[Dict[str, Any]]:
    """
    Define relationships between metadata fields for cross-validation.
    
    Returns:
        List of field relationship definitions
    """
    return [
        {
            "fields": ["invoice_date", "due_date"],
            "relationship": "date_sequence",
            "description": "Due date should be after invoice date"
        },
        {
            "fields": ["subtotal", "tax_amount", "total_amount"],
            "relationship": "sum",
            "description": "Subtotal + tax should approximately equal total"
        },
        {
            "fields": ["invoice_number", "po_number"],
            "relationship": "distinct",
            "description": "Invoice number and PO number should be different"
        },
        {
            "fields": ["vendor_name", "vendor_address"],
            "relationship": "co_presence",
            "description": "If vendor name exists, vendor address likely exists too"
        }
    ]

def define_validation_rules() -> Dict[str, Dict[str, Any]]:
    """
    Define validation rules for metadata fields.
    
    Returns:
        Dictionary of field validation rules
    """
    return {
        "invoice_number": {
            "min_length": 3,
            "max_length": 20,
            "pattern": r"[A-Za-z0-9\-]+"
        },
        "po_number": {
            "min_length": 3,
            "max_length": 20,
            "pattern": r"[A-Za-z0-9\-]+"
        },
        "total_amount": {
            "min_value": 0,
            "pattern": r"\d+(\.\d{1,2})?"
        },
        "email": {
            "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        },
        "phone": {
            "pattern": r"[\d\+\-\(\)\s]+"
        }
    }
