"""
Enhanced Confidence Framework for Document Categorization and Metadata Extraction

This module provides advanced confidence calculation mechanisms for both document categorization
and metadata extraction, offering more robust, explainable, and accurate confidence scores.

Features:
- Advanced multi-factor confidence model for document categorization
- Field-specific confidence calculation for metadata extraction
- Cross-validation mechanisms for metadata confidence
- Calibration based on historical performance
- Visualization utilities for confidence scores
"""

import streamlit as st
import logging
import re
import os
import json
import numpy as np
import pandas as pd
import altair as alt
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Document Categorization Confidence Framework ----

def calculate_enhanced_confidence(
    ai_reported_confidence: float,
    document_features: Dict[str, Any],
    assigned_category: str,
    reasoning: str,
    valid_categories: List[str],
    document_text: Optional[str] = None,
    category_descriptions: Optional[Dict[str, str]] = None
) -> Dict[str, float]:
    """
    Calculate an enhanced multi-factor confidence score with more sophisticated factors.
    
    Args:
        ai_reported_confidence: Raw confidence score from the AI model
        document_features: Features extracted from the document
        assigned_category: The category assigned by the AI model
        reasoning: The reasoning provided by the AI model
        valid_categories: List of valid category names
        document_text: Optional text content of the document for semantic analysis
        category_descriptions: Optional descriptions of categories for semantic matching
        
    Returns:
        Dictionary containing individual factor scores and overall confidence
    """
    factors = {
        "semantic_alignment": 0.0,
        "feature_correlation": 0.0,
        "reasoning_coherence": 0.0,
        "exclusion_confidence": 0.0,
        "historical_performance": 0.0,
        "overall": 0.0
    }
    
    # 1. Semantic Alignment Score
    # How well the document content aligns with the assigned category
    if document_text and category_descriptions and assigned_category in category_descriptions:
        # Calculate semantic similarity between document text and category description
        # For now, use a simplified keyword-based approach
        category_desc = category_descriptions[assigned_category].lower()
        doc_text_lower = document_text.lower()
        
        # Extract key terms from category description
        key_terms = set([term.strip() for term in re.split(r'[,\s]', category_desc) if len(term.strip()) > 3])
        
        # Count matches in document text
        matches = sum(1 for term in key_terms if term in doc_text_lower)
        if key_terms:
            match_ratio = matches / len(key_terms)
            factors["semantic_alignment"] = min(1.0, match_ratio * 1.5)  # Scale up slightly, cap at 1.0
        else:
            factors["semantic_alignment"] = 0.5  # Default if no key terms found
    else:
        # Fallback if text or descriptions not available
        factors["semantic_alignment"] = ai_reported_confidence * 0.9  # Slightly discount AI confidence
    
    # 2. Feature Correlation Score
    # How well document features correlate with typical features for the category
    ext = document_features.get("file_extension", "").lower()
    size_kb = document_features.get("file_size_kb", 0)
    
    # Enhanced feature matching based on document type
    feature_score = 0.5  # Default
    
    # More sophisticated feature matching logic
    if assigned_category == "Invoices":
        if ext in [".pdf", ".docx", ".xlsx"]:
            feature_score = 0.7
            if size_kb < 1024:  # Typical invoices are smaller
                feature_score += 0.1
            # Check for invoice-related keywords in reasoning
            if any(kw in reasoning.lower() for kw in ["invoice", "bill", "payment", "amount", "total", "due"]):
                feature_score += 0.1
        else:
            feature_score = 0.4
    
    elif assigned_category == "Sales Contract":
        if ext in [".pdf", ".docx"]:
            feature_score = 0.7
            if size_kb > 100:  # Contracts tend to be larger
                feature_score += 0.1
            # Check for contract-related keywords
            if any(kw in reasoning.lower() for kw in ["contract", "agreement", "terms", "parties", "signed"]):
                feature_score += 0.1
        else:
            feature_score = 0.4
    
    elif assigned_category == "Financial Report":
        if ext in [".pdf", ".xlsx", ".csv"]:
            feature_score = 0.7
            if size_kb > 200:  # Financial reports tend to be larger
                feature_score += 0.1
            # Check for financial report keywords
            if any(kw in reasoning.lower() for kw in ["financial", "report", "quarter", "annual", "balance", "income"]):
                feature_score += 0.1
        else:
            feature_score = 0.4
    
    elif assigned_category == "Tax":
        if ext in [".pdf", ".xlsx"]:
            feature_score = 0.7
            # Check for tax-related keywords
            if any(kw in reasoning.lower() for kw in ["tax", "return", "irs", "deduction", "income", "filing"]):
                feature_score += 0.2
        else:
            feature_score = 0.4
    
    elif assigned_category == "Employment Contract":
        if ext in [".pdf", ".docx"]:
            feature_score = 0.7
            # Check for employment-related keywords
            if any(kw in reasoning.lower() for kw in ["employment", "salary", "position", "employee", "employer"]):
                feature_score += 0.2
        else:
            feature_score = 0.4
    
    elif assigned_category == "PII":
        # PII can be in many formats, so don't penalize based on extension
        feature_score = 0.6
        # Check for PII-related keywords
        if any(kw in reasoning.lower() for kw in ["personal", "information", "identity", "ssn", "address", "phone"]):
            feature_score += 0.2
    
    # Cap at 1.0
    factors["feature_correlation"] = min(1.0, feature_score)
    
    # 3. Reasoning Coherence Score
    # Evaluate the quality and coherence of the AI's reasoning
    reasoning_score = 0.4  # Default
    
    # Length-based assessment
    if len(reasoning) > 200:
        reasoning_score += 0.2
    elif len(reasoning) > 100:
        reasoning_score += 0.1
    
    # Check for evidence mentions
    evidence_keywords = ["evidence", "indicates", "shows", "contains", "demonstrates", "example", "instance"]
    evidence_count = sum(1 for kw in evidence_keywords if kw in reasoning.lower())
    reasoning_score += min(0.2, evidence_count * 0.05)  # Up to 0.2 for evidence
    
    # Check for structured reasoning (e.g., points, numbered items)
    if re.search(r'\d+\.\s+|\*\s+|-\s+', reasoning):
        reasoning_score += 0.1
    
    # Check for category-specific terminology
    category_terms = assigned_category.lower().split()
    category_term_count = sum(1 for term in category_terms if term in reasoning.lower())
    reasoning_score += min(0.1, category_term_count * 0.05)  # Up to 0.1 for category terms
    
    factors["reasoning_coherence"] = min(1.0, reasoning_score)
    
    # 4. Exclusion Confidence Score
    # How confidently the model excluded other categories
    exclusion_score = 0.5  # Default
    
    # Check if reasoning explicitly mentions why other categories don't apply
    other_categories = [cat for cat in valid_categories if cat != assigned_category]
    exclusion_mentions = sum(1 for cat in other_categories if cat.lower() in reasoning.lower())
    
    if exclusion_mentions > 0:
        exclusion_score += min(0.3, exclusion_mentions * 0.1)  # Up to 0.3 for mentioning other categories
    
    # If AI confidence is very high, boost exclusion confidence
    if ai_reported_confidence > 0.9:
        exclusion_score += 0.1
    
    # If "Other" category, reduce exclusion confidence
    if assigned_category == "Other":
        exclusion_score *= 0.7
    
    factors["exclusion_confidence"] = min(1.0, exclusion_score)
    
    # 5. Historical Performance Score
    # Based on historical accuracy for this category (placeholder implementation)
    # In a real implementation, this would use stored historical data
    historical_score = 0.7  # Default assumption of good performance
    
    # Adjust based on category (placeholder logic)
    if assigned_category == "Other":
        historical_score = 0.5  # "Other" category tends to be less reliable
    elif assigned_category in ["Invoices", "Tax"]:
        historical_score = 0.8  # Assume these are commonly accurate
    
    factors["historical_performance"] = historical_score
    
    # Calculate Overall Confidence (Weighted average)
    weights = {
        "semantic_alignment": 0.25,
        "feature_correlation": 0.20,
        "reasoning_coherence": 0.25,
        "exclusion_confidence": 0.15,
        "historical_performance": 0.15
    }
    
    overall_score = sum(factors[key] * weights[key] for key in weights)
    factors["overall"] = min(1.0, max(0.0, overall_score))
    
    # Also include the original AI reported confidence for reference
    factors["ai_reported"] = ai_reported_confidence
    
    return factors

def apply_category_specific_calibration(
    category: str, 
    confidence: float,
    document_features: Dict[str, Any]
) -> float:
    """
    Apply category-specific calibration to adjust confidence scores based on
    historical performance patterns for different document types.
    
    Args:
        category: Document category
        confidence: Raw confidence score
        document_features: Features of the document
        
    Returns:
        Calibrated confidence score
    """
    # Start with the original confidence
    calibrated = confidence
    
    # Apply category-specific adjustments
    if category == "Other":
        # "Other" category tends to be overconfident
        calibrated *= 0.85
    elif category == "Invoices":
        # Invoices are usually well-recognized, but check extension
        ext = document_features.get("file_extension", "").lower()
        if ext in [".pdf", ".docx", ".xlsx"]:
            calibrated = min(1.0, calibrated * 1.05)  # Slight boost
        else:
            calibrated *= 0.95  # Slight reduction
    elif category == "PII":
        # PII detection can be tricky, be more conservative
        calibrated *= 0.9
    elif category == "Financial Report":
        # Financial reports can be complex, adjust based on size
        size_kb = document_features.get("file_size_kb", 0)
        if size_kb < 50:  # Very small for a financial report
            calibrated *= 0.9
    
    # Ensure the calibrated score is within valid range
    return min(1.0, max(0.0, calibrated))

def get_confidence_explanation(factors: Dict[str, float], category: str) -> Dict[str, str]:
    """
    Generate human-readable explanations for each confidence factor.
    
    Args:
        factors: Dictionary of confidence factors
        category: Document category
        
    Returns:
        Dictionary mapping factor names to explanations
    """
    explanations = {}
    
    # Semantic Alignment
    semantic_score = factors.get("semantic_alignment", 0.0)
    if semantic_score > 0.8:
        explanations["semantic_alignment"] = f"Document content strongly aligns with '{category}' category characteristics."
    elif semantic_score > 0.6:
        explanations["semantic_alignment"] = f"Document content moderately aligns with '{category}' category characteristics."
    else:
        explanations["semantic_alignment"] = f"Document content shows limited alignment with '{category}' category characteristics."
    
    # Feature Correlation
    feature_score = factors.get("feature_correlation", 0.0)
    if feature_score > 0.8:
        explanations["feature_correlation"] = f"Document features (file type, size) strongly match typical '{category}' documents."
    elif feature_score > 0.6:
        explanations["feature_correlation"] = f"Document features partially match typical '{category}' documents."
    else:
        explanations["feature_correlation"] = f"Document features show limited correlation with typical '{category}' documents."
    
    # Reasoning Coherence
    reasoning_score = factors.get("reasoning_coherence", 0.0)
    if reasoning_score > 0.8:
        explanations["reasoning_coherence"] = "AI provided detailed, evidence-based reasoning for this categorization."
    elif reasoning_score > 0.6:
        explanations["reasoning_coherence"] = "AI provided adequate reasoning for this categorization."
    else:
        explanations["reasoning_coherence"] = "AI provided limited or weak reasoning for this categorization."
    
    # Exclusion Confidence
    exclusion_score = factors.get("exclusion_confidence", 0.0)
    if exclusion_score > 0.8:
        explanations["exclusion_confidence"] = "AI confidently excluded other potential categories."
    elif exclusion_score > 0.6:
        explanations["exclusion_confidence"] = "AI moderately justified why other categories don't apply."
    else:
        explanations["exclusion_confidence"] = "AI provided limited justification for excluding other categories."
    
    # Historical Performance
    historical_score = factors.get("historical_performance", 0.0)
    if historical_score > 0.8:
        explanations["historical_performance"] = f"AI has historically performed well on '{category}' documents."
    elif historical_score > 0.6:
        explanations["historical_performance"] = f"AI has moderate historical accuracy on '{category}' documents."
    else:
        explanations["historical_performance"] = f"AI has limited historical accuracy on '{category}' documents."
    
    # AI Reported
    ai_score = factors.get("ai_reported", 0.0)
    if ai_score > 0.8:
        explanations["ai_reported"] = "AI reported high confidence in its categorization."
    elif ai_score > 0.6:
        explanations["ai_reported"] = "AI reported moderate confidence in its categorization."
    else:
        explanations["ai_reported"] = "AI reported low confidence in its categorization."
    
    # Overall
    overall_score = factors.get("overall", 0.0)
    if overall_score > 0.85:
        explanations["overall"] = "Overall high confidence in document categorization."
    elif overall_score > 0.7:
        explanations["overall"] = "Overall good confidence in document categorization."
    elif overall_score > 0.5:
        explanations["overall"] = "Overall moderate confidence in document categorization."
    else:
        explanations["overall"] = "Overall low confidence in document categorization."
    
    return explanations

def create_confidence_visualization(factors: Dict[str, float]) -> alt.Chart:
    """
    Create an Altair visualization of confidence factors.
    
    Args:
        factors: Dictionary of confidence factors
        
    Returns:
        Altair chart object
    """
    # Prepare data for visualization
    factor_names = {
        "semantic_alignment": "Semantic Alignment",
        "feature_correlation": "Feature Correlation",
        "reasoning_coherence": "Reasoning Coherence",
        "exclusion_confidence": "Exclusion Confidence",
        "historical_performance": "Historical Performance",
        "ai_reported": "AI Reported",
        "overall": "Overall"
    }
    
    # Filter out any non-numeric values and ensure we have all expected keys
    numeric_factors = {k: float(v) for k, v in factors.items() if k in factor_names and isinstance(v, (int, float))}
    
    # Create dataframe for visualization
    data = []
    for factor_key, factor_value in numeric_factors.items():
        if factor_key != "overall":  # We'll handle overall separately
            data.append({
                "Factor": factor_names.get(factor_key, factor_key),
                "Score": factor_value,
                "Color": get_confidence_color_name(factor_value)
            })
    
    df = pd.DataFrame(data)
    
    # Create the chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Factor:N', sort=None),
        color=alt.Color('Color:N', scale=alt.Scale(
            domain=['Low', 'Medium', 'High'],
            range=['#FF5252', '#FFA726', '#66BB6A']
        ), legend=None),
        tooltip=['Factor', 'Score']
    ).properties(
        title='Confidence Factor Breakdown',
        width=400,
        height=200
    )
    
    return chart

def get_confidence_color_name(score: float) -> str:
    """Get color name based on confidence score."""
    if score >= 0.7:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"

def get_confidence_color_hex(score: float) -> str:
    """Get hex color code based on confidence score."""
    if score >= 0.7:
        return "#66BB6A"  # Green
    elif score >= 0.5:
        return "#FFA726"  # Orange
    else:
        return "#FF5252"  # Red

# ---- Metadata Extraction Confidence Framework ----

def calculate_metadata_field_confidence(
    field_name: str,
    field_type: str,
    extracted_value: Any,
    ai_confidence: str,
    document_text: Optional[str] = None,
    field_definition: Optional[Dict[str, Any]] = None,
    validation_rules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate enhanced confidence for a metadata field extraction.
    
    Args:
        field_name: Name of the metadata field
        field_type: Type of the field (string, date, number, etc.)
        extracted_value: Value extracted by the AI
        ai_confidence: AI-reported confidence (High, Medium, Low)
        document_text: Optional document text for validation
        field_definition: Optional field definition with expected format
        validation_rules: Optional validation rules for the field
        
    Returns:
        Dictionary with confidence score and factors
    """
    # Convert categorical AI confidence to numeric
    ai_confidence_map = {"High": 0.9, "Medium": 0.6, "Low": 0.3}
    ai_confidence_numeric = ai_confidence_map.get(ai_confidence, 0.3)
    
    # Initialize confidence factors
    factors = {
        "ai_reported": ai_confidence_numeric,
        "format_validity": 0.0,
        "content_plausibility": 0.0,
        "extraction_consistency": 0.0,
        "overall": 0.0
    }
    
    # 1. Format Validity - Check if the extracted value matches expected format
    format_score = 0.5  # Default
    
    if extracted_value is None:
        format_score = 0.0
    elif field_type == "string":
        if isinstance(extracted_value, str):
            format_score = 0.8
            # Check if string is not just whitespace
            if extracted_value.strip():
                format_score = 0.9
        else:
            format_score = 0.3
    elif field_type == "date":
        # Check if it's a valid date format
        if isinstance(extracted_value, str):
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',   # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',   # MM-DD-YYYY
                r'\d{1,2}\s+[A-Za-z]+\s+\d{4}'  # DD Month YYYY
            ]
            if any(re.match(pattern, extracted_value) for pattern in date_patterns):
                format_score = 0.9
            else:
                format_score = 0.4
        else:
            format_score = 0.3
    elif field_type == "number":
        # Check if it's a valid number
        if isinstance(extracted_value, (int, float)):
            format_score = 0.9
        elif isinstance(extracted_value, str):
            # Check if string can be converted to number
            try:
                float(extracted_value.replace(',', ''))
                format_score = 0.8
            except ValueError:
                format_score = 0.2
        else:
            format_score = 0.2
    elif field_type == "enum" and field_definition and "options" in field_definition:
        # Check if value is in allowed options
        options = field_definition["options"]
        if extracted_value in options:
            format_score = 1.0
        else:
            # Check for close matches
            if isinstance(extracted_value, str) and any(opt.lower() == extracted_value.lower() for opt in options):
                format_score = 0.9
            else:
                format_score = 0.2
    
    factors["format_validity"] = format_score
    
    # 2. Content Plausibility - Check if the value is plausible for the field
    plausibility_score = 0.5  # Default
    
    # Field-specific plausibility checks
    if "invoice" in field_name.lower() and field_type == "string":
        # Invoice numbers often have specific patterns
        if isinstance(extracted_value, str):
            if re.match(r'(INV|inv|Invoice|INVOICE)[-\s]?\d+', extracted_value):
                plausibility_score = 0.9
            elif re.match(r'\d{5,10}', extracted_value):  # Just digits
                plausibility_score = 0.7
    elif "date" in field_name.lower() and field_type == "date":
        # Check if date is within reasonable range (not future, not too old)
        if isinstance(extracted_value, str):
            try:
                # Try to parse the date
                date_obj = None
                date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y"]
                
                for fmt in date_formats:
                    try:
                        date_obj = datetime.strptime(extracted_value, fmt)
                        break
                    except ValueError:
                        continue
                
                if date_obj:
                    # Check if date is in reasonable range
                    now = datetime.now()
                    if date_obj <= now and date_obj.year > 1900:
                        plausibility_score = 0.9
                    elif date_obj.year > now.year:
                        plausibility_score = 0.3  # Future date
                    else:
                        plausibility_score = 0.6  # Very old date
            except Exception:
                plausibility_score = 0.4
    elif "amount" in field_name.lower() and field_type == "number":
        # Check if amount is a reasonable value
        try:
            if isinstance(extracted_value, (int, float)):
                amount = extracted_value
            elif isinstance(extracted_value, str):
                # Remove currency symbols and commas
                cleaned = re.sub(r'[^\d.-]', '', extracted_value)
                amount = float(cleaned)
                
            # Check if amount is reasonable (not negative, not too large)
            if amount >= 0 and amount < 1000000000:  # Less than a billion
                plausibility_score = 0.8
            elif amount < 0:
                plausibility_score = 0.3  # Negative amount
            else:
                plausibility_score = 0.4  # Very large amount
        except Exception:
            plausibility_score = 0.4
    elif "name" in field_name.lower() and field_type == "string":
        # Check if name looks reasonable
        if isinstance(extracted_value, str):
            if len(extracted_value.split()) >= 2:  # At least two words
                plausibility_score = 0.8
            elif len(extracted_value) > 2:  # At least some characters
                plausibility_score = 0.6
            else:
                plausibility_score = 0.3
    
    # Apply custom validation rules if provided
    if validation_rules and field_name in validation_rules:
        rule = validation_rules[field_name]
        if "min_length" in rule and isinstance(extracted_value, str):
            if len(extracted_value) >= rule["min_length"]:
                plausibility_score = min(1.0, plausibility_score + 0.1)
            else:
                plausibility_score *= 0.8
        if "max_length" in rule and isinstance(extracted_value, str):
            if len(extracted_value) <= rule["max_length"]:
                plausibility_score = min(1.0, plausibility_score + 0.1)
            else:
                plausibility_score *= 0.8
        if "pattern" in rule and isinstance(extracted_value, str):
            if re.match(rule["pattern"], extracted_value):
                plausibility_score = min(1.0, plausibility_score + 0.2)
            else:
                plausibility_score *= 0.7
    
    factors["content_plausibility"] = plausibility_score
    
    # 3. Extraction Consistency - Check if the value is consistently found in the document
    consistency_score = 0.5  # Default
    
    if document_text and isinstance(extracted_value, str) and extracted_value.strip():
        # Check if the extracted value appears in the document
        if extracted_value in document_text:
            # Count occurrences
            occurrences = document_text.count(extracted_value)
            if occurrences == 1:
                consistency_score = 0.9  # Unique occurrence is good
            elif occurrences > 1:
                consistency_score = 0.7  # Multiple occurrences create ambiguity
            
            # Check if it appears in a context that matches the field name
            field_context = field_name.replace("_", " ").lower()
            context_window = 100  # Characters to check before/after
            
            # Find all positions of the extracted value
            positions = [m.start() for m in re.finditer(re.escape(extracted_value), document_text)]
            
            for pos in positions:
                # Get context around the value
                start = max(0, pos - context_window)
                end = min(len(document_text), pos + len(extracted_value) + context_window)
                context = document_text[start:end].lower()
                
                # Check if field name or related terms appear in context
                if field_context in context or any(term in context for term in field_context.split()):
                    consistency_score = min(1.0, consistency_score + 0.2)
                    break
        else:
            # Value doesn't appear exactly in text
            consistency_score = 0.3
            
            # Check if parts of the value appear (for multi-word values)
            if len(extracted_value.split()) > 1:
                parts_found = sum(1 for part in extracted_value.split() if part in document_text)
                if parts_found > 0:
                    consistency_score = 0.3 + (0.4 * parts_found / len(extracted_value.split()))
    
    factors["extraction_consistency"] = consistency_score
    
    # Calculate Overall Confidence (Weighted average)
    weights = {
        "ai_reported": 0.4,
        "format_validity": 0.3,
        "content_plausibility": 0.2,
        "extraction_consistency": 0.1
    }
    
    overall_score = sum(factors[key] * weights[key] for key in weights)
    factors["overall"] = min(1.0, max(0.0, overall_score))
    
    # Convert numeric score back to categorical for compatibility
    categorical_confidence = "Low"
    if factors["overall"] >= 0.8:
        categorical_confidence = "High"
    elif factors["overall"] >= 0.5:
        categorical_confidence = "Medium"
    
    return {
        "numeric_confidence": factors["overall"],
        "categorical_confidence": categorical_confidence,
        "factors": factors,
        "original_ai_confidence": ai_confidence
    }

def validate_metadata_cross_field(
    extracted_fields: Dict[str, Any],
    confidence_scores: Dict[str, Dict[str, Any]],
    field_relationships: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Validate metadata extraction across related fields and adjust confidence accordingly.
    
    Args:
        extracted_fields: Dictionary of extracted field values
        confidence_scores: Dictionary of confidence scores for each field
        field_relationships: Optional list of field relationship definitions
        
    Returns:
        Updated confidence scores
    """
    if not field_relationships:
        # Default relationships if none provided
        field_relationships = [
            {
                "fields": ["invoice_date", "due_date"],
                "relationship": "date_sequence",
                "description": "Due date should be after invoice date"
            },
            {
                "fields": ["subtotal", "tax_amount", "total_amount"],
                "relationship": "sum",
                "description": "Subtotal + tax should approximately equal total"
            }
        ]
    
    # Copy confidence scores to avoid modifying the original
    updated_scores = {k: v.copy() if isinstance(v, dict) else v for k, v in confidence_scores.items()}
    
    # Process each relationship
    for relationship in field_relationships:
        rel_type = relationship.get("relationship")
        fields = relationship.get("fields", [])
        
        # Skip if any field is missing
        if not all(field in extracted_fields for field in fields):
            continue
        
        # Get values for all fields in this relationship
        values = {field: extracted_fields[field] for field in fields}
        
        # Apply relationship-specific validation
        if rel_type == "date_sequence":
            # Check if dates are in correct sequence
            try:
                date1 = values[fields[0]]
                date2 = values[fields[1]]
                
                # Parse dates if they're strings
                if isinstance(date1, str) and isinstance(date2, str):
                    # Try common formats
                    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]
                    date1_obj = None
                    date2_obj = None
                    
                    for fmt in formats:
                        try:
                            date1_obj = datetime.strptime(date1, fmt)
                            break
                        except ValueError:
                            continue
                    
                    for fmt in formats:
                        try:
                            date2_obj = datetime.strptime(date2, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if date1_obj and date2_obj:
                        if date2_obj >= date1_obj:
                            # Dates are in correct sequence, boost confidence
                            for field in fields:
                                if field in updated_scores:
                                    factors = updated_scores[field].get("factors", {})
                                    factors["content_plausibility"] = min(1.0, factors.get("content_plausibility", 0.5) + 0.1)
                                    
                                    # Recalculate overall
                                    weights = {
                                        "ai_reported": 0.4,
                                        "format_validity": 0.3,
                                        "content_plausibility": 0.2,
                                        "extraction_consistency": 0.1
                                    }
                                    overall = sum(factors.get(k, 0) * weights[k] for k in weights)
                                    factors["overall"] = min(1.0, max(0.0, overall))
                                    
                                    # Update categorical confidence
                                    if factors["overall"] >= 0.8:
                                        updated_scores[field]["categorical_confidence"] = "High"
                                    elif factors["overall"] >= 0.5:
                                        updated_scores[field]["categorical_confidence"] = "Medium"
                                    else:
                                        updated_scores[field]["categorical_confidence"] = "Low"
                                    
                                    updated_scores[field]["numeric_confidence"] = factors["overall"]
                                    updated_scores[field]["factors"] = factors
                        else:
                            # Dates are in wrong sequence, reduce confidence
                            for field in fields:
                                if field in updated_scores:
                                    factors = updated_scores[field].get("factors", {})
                                    factors["content_plausibility"] = max(0.0, factors.get("content_plausibility", 0.5) - 0.2)
                                    
                                    # Recalculate overall
                                    weights = {
                                        "ai_reported": 0.4,
                                        "format_validity": 0.3,
                                        "content_plausibility": 0.2,
                                        "extraction_consistency": 0.1
                                    }
                                    overall = sum(factors.get(k, 0) * weights[k] for k in weights)
                                    factors["overall"] = min(1.0, max(0.0, overall))
                                    
                                    # Update categorical confidence
                                    if factors["overall"] >= 0.8:
                                        updated_scores[field]["categorical_confidence"] = "High"
                                    elif factors["overall"] >= 0.5:
                                        updated_scores[field]["categorical_confidence"] = "Medium"
                                    else:
                                        updated_scores[field]["categorical_confidence"] = "Low"
                                    
                                    updated_scores[field]["numeric_confidence"] = factors["overall"]
                                    updated_scores[field]["factors"] = factors
            except Exception as e:
                logger.warning(f"Error validating date sequence: {str(e)}")
        
        elif rel_type == "sum":
            # Check if sum relationship holds
            try:
                # Convert values to float if they're strings
                numeric_values = {}
                for field, value in values.items():
                    if isinstance(value, (int, float)):
                        numeric_values[field] = float(value)
                    elif isinstance(value, str):
                        # Remove currency symbols and commas
                        cleaned = re.sub(r'[^\d.-]', '', value)
                        numeric_values[field] = float(cleaned)
                
                # Check if subtotal + tax ≈ total
                if len(fields) == 3:  # Assuming [subtotal, tax, total]
                    subtotal = numeric_values[fields[0]]
                    tax = numeric_values[fields[1]]
                    total = numeric_values[fields[2]]
                    
                    # Allow for small rounding differences
                    if abs((subtotal + tax) - total) < 0.02 * total:
                        # Sum relationship holds, boost confidence
                        for field in fields:
                            if field in updated_scores:
                                factors = updated_scores[field].get("factors", {})
                                factors["content_plausibility"] = min(1.0, factors.get("content_plausibility", 0.5) + 0.15)
                                
                                # Recalculate overall
                                weights = {
                                    "ai_reported": 0.4,
                                    "format_validity": 0.3,
                                    "content_plausibility": 0.2,
                                    "extraction_consistency": 0.1
                                }
                                overall = sum(factors.get(k, 0) * weights[k] for k in weights)
                                factors["overall"] = min(1.0, max(0.0, overall))
                                
                                # Update categorical confidence
                                if factors["overall"] >= 0.8:
                                    updated_scores[field]["categorical_confidence"] = "High"
                                elif factors["overall"] >= 0.5:
                                    updated_scores[field]["categorical_confidence"] = "Medium"
                                else:
                                    updated_scores[field]["categorical_confidence"] = "Low"
                                
                                updated_scores[field]["numeric_confidence"] = factors["overall"]
                                updated_scores[field]["factors"] = factors
                    else:
                        # Sum relationship doesn't hold, reduce confidence
                        for field in fields:
                            if field in updated_scores:
                                factors = updated_scores[field].get("factors", {})
                                factors["content_plausibility"] = max(0.0, factors.get("content_plausibility", 0.5) - 0.25)
                                
                                # Recalculate overall
                                weights = {
                                    "ai_reported": 0.4,
                                    "format_validity": 0.3,
                                    "content_plausibility": 0.2,
                                    "extraction_consistency": 0.1
                                }
                                overall = sum(factors.get(k, 0) * weights[k] for k in weights)
                                factors["overall"] = min(1.0, max(0.0, overall))
                                
                                # Update categorical confidence
                                if factors["overall"] >= 0.8:
                                    updated_scores[field]["categorical_confidence"] = "High"
                                elif factors["overall"] >= 0.5:
                                    updated_scores[field]["categorical_confidence"] = "Medium"
                                else:
                                    updated_scores[field]["categorical_confidence"] = "Low"
                                
                                updated_scores[field]["numeric_confidence"] = factors["overall"]
                                updated_scores[field]["factors"] = factors
            except Exception as e:
                logger.warning(f"Error validating sum relationship: {str(e)}")
    
    return updated_scores

def get_metadata_confidence_explanation(confidence_result: Dict[str, Any], field_name: str) -> Dict[str, str]:
    """
    Generate human-readable explanations for metadata confidence factors.
    
    Args:
        confidence_result: Confidence calculation result
        field_name: Name of the metadata field
        
    Returns:
        Dictionary mapping factor names to explanations
    """
    explanations = {}
    factors = confidence_result.get("factors", {})
    
    # AI Reported
    ai_score = factors.get("ai_reported", 0.0)
    if ai_score > 0.8:
        explanations["ai_reported"] = f"AI reported high confidence in extracting '{field_name}'."
    elif ai_score > 0.5:
        explanations["ai_reported"] = f"AI reported moderate confidence in extracting '{field_name}'."
    else:
        explanations["ai_reported"] = f"AI reported low confidence in extracting '{field_name}'."
    
    # Format Validity
    format_score = factors.get("format_validity", 0.0)
    if format_score > 0.8:
        explanations["format_validity"] = f"The extracted value for '{field_name}' has the correct format."
    elif format_score > 0.5:
        explanations["format_validity"] = f"The extracted value for '{field_name}' has a partially valid format."
    else:
        explanations["format_validity"] = f"The extracted value for '{field_name}' does not match the expected format."
    
    # Content Plausibility
    plausibility_score = factors.get("content_plausibility", 0.0)
    if plausibility_score > 0.8:
        explanations["content_plausibility"] = f"The value for '{field_name}' is highly plausible for this field type."
    elif plausibility_score > 0.5:
        explanations["content_plausibility"] = f"The value for '{field_name}' is moderately plausible for this field type."
    else:
        explanations["content_plausibility"] = f"The value for '{field_name}' has low plausibility for this field type."
    
    # Extraction Consistency
    consistency_score = factors.get("extraction_consistency", 0.0)
    if consistency_score > 0.8:
        explanations["extraction_consistency"] = f"The value for '{field_name}' was consistently found in the document."
    elif consistency_score > 0.5:
        explanations["extraction_consistency"] = f"The value for '{field_name}' was found in the document with some ambiguity."
    else:
        explanations["extraction_consistency"] = f"The value for '{field_name}' was not clearly found in the document."
    
    # Overall
    overall_score = factors.get("overall", 0.0)
    if overall_score > 0.8:
        explanations["overall"] = f"Overall high confidence in the extraction of '{field_name}'."
    elif overall_score > 0.5:
        explanations["overall"] = f"Overall moderate confidence in the extraction of '{field_name}'."
    else:
        explanations["overall"] = f"Overall low confidence in the extraction of '{field_name}'."
    
    return explanations

def create_metadata_confidence_visualization(confidence_result: Dict[str, Any]) -> alt.Chart:
    """
    Create an Altair visualization of metadata confidence factors.
    
    Args:
        confidence_result: Confidence calculation result
        
    Returns:
        Altair chart object
    """
    # Prepare data for visualization
    factors = confidence_result.get("factors", {})
    
    factor_names = {
        "ai_reported": "AI Reported",
        "format_validity": "Format Validity",
        "content_plausibility": "Content Plausibility",
        "extraction_consistency": "Extraction Consistency",
        "overall": "Overall"
    }
    
    # Filter out any non-numeric values and ensure we have all expected keys
    numeric_factors = {k: float(v) for k, v in factors.items() if k in factor_names and isinstance(v, (int, float))}
    
    # Create dataframe for visualization
    data = []
    for factor_key, factor_value in numeric_factors.items():
        if factor_key != "overall":  # We'll handle overall separately
            data.append({
                "Factor": factor_names.get(factor_key, factor_key),
                "Score": factor_value,
                "Color": get_confidence_color_name(factor_value)
            })
    
    df = pd.DataFrame(data)
    
    # Create the chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Factor:N', sort=None),
        color=alt.Color('Color:N', scale=alt.Scale(
            domain=['Low', 'Medium', 'High'],
            range=['#FF5252', '#FFA726', '#66BB6A']
        ), legend=None),
        tooltip=['Factor', 'Score']
    ).properties(
        title='Metadata Confidence Factor Breakdown',
        width=400,
        height=150
    )
    
    return chart
