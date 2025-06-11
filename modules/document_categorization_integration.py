"""
Integration module for enhanced confidence framework with document categorization.

This module integrates the enhanced confidence framework with the existing
document categorization functionality, ensuring backward compatibility.
"""

import streamlit as st
import logging
import pandas as pd
import altair as alt
from typing import Dict, Any, List, Optional, Tuple

# Import the enhanced confidence framework
from modules.enhanced_confidence_framework import (
    calculate_enhanced_confidence,
    apply_category_specific_calibration,
    get_confidence_explanation,
    create_confidence_visualization,
    get_confidence_color_name,
    get_confidence_color_hex
)

# Import existing document categorization utilities
from modules.document_categorization_utils import (
    calculate_multi_factor_confidence,
    apply_confidence_calibration,
    apply_confidence_thresholds
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_with_enhanced_confidence(
    categorization_result: Dict[str, Any],
    document_features: Dict[str, Any],
    valid_categories: List[str],
    document_text: Optional[str] = None,
    category_descriptions: Optional[Dict[str, str]] = None,
    use_enhanced: bool = True
) -> Dict[str, Any]:
    """
    Process categorization results with enhanced confidence framework.
    
    Args:
        categorization_result: Original categorization result
        document_features: Features extracted from the document
        valid_categories: List of valid category names
        document_text: Optional text content of the document
        category_descriptions: Optional descriptions of categories
        use_enhanced: Whether to use enhanced confidence framework
        
    Returns:
        Updated categorization result with confidence scores
    """
    # Extract basic information
    category = categorization_result.get("document_type", "Other")
    ai_confidence = categorization_result.get("confidence", 0.0)
    reasoning = categorization_result.get("reasoning", "")
    
    # Process with appropriate confidence framework
    if use_enhanced:
        # Use enhanced confidence framework
        confidence_factors = calculate_enhanced_confidence(
            ai_reported_confidence=ai_confidence,
            document_features=document_features,
            assigned_category=category,
            reasoning=reasoning,
            valid_categories=valid_categories,
            document_text=document_text,
            category_descriptions=category_descriptions
        )
        
        # Apply category-specific calibration
        calibrated_confidence = apply_category_specific_calibration(
            category=category,
            confidence=confidence_factors["overall"],
            document_features=document_features
        )
        
        # Generate explanations
        explanations = get_confidence_explanation(confidence_factors, category)
        
        # Update result with enhanced confidence information
        result = categorization_result.copy()
        result["enhanced_confidence_factors"] = confidence_factors
        result["calibrated_confidence"] = calibrated_confidence
        result["confidence_explanations"] = explanations
        result["confidence_framework"] = "enhanced"
        
        # For backward compatibility, also include multi_factor_confidence
        result["multi_factor_confidence"] = {
            "ai_reported": confidence_factors["ai_reported"],
            "response_quality": confidence_factors["reasoning_coherence"],
            "category_specificity": confidence_factors["exclusion_confidence"],
            "reasoning_quality": confidence_factors["reasoning_coherence"],
            "document_features_match": confidence_factors["feature_correlation"],
            "overall": confidence_factors["overall"]
        }
    else:
        # Use original confidence framework for backward compatibility
        multi_factor_confidence = calculate_multi_factor_confidence(
            ai_reported_confidence=ai_confidence,
            document_features=document_features,
            assigned_category=category,
            reasoning=reasoning,
            valid_categories=valid_categories
        )
        
        calibrated_confidence = apply_confidence_calibration(
            category=category,
            confidence=multi_factor_confidence["overall"]
        )
        
        # Update result with original confidence information
        result = categorization_result.copy()
        result["multi_factor_confidence"] = multi_factor_confidence
        result["calibrated_confidence"] = calibrated_confidence
        result["confidence_framework"] = "original"
    
    return result

def display_enhanced_confidence_visualization(result: Dict[str, Any], show_explanations: bool = True) -> None:
    """
    Display enhanced confidence visualization in Streamlit.
    
    Args:
        result: Categorization result with confidence information
        show_explanations: Whether to show explanations
    """
    if "enhanced_confidence_factors" in result:
        # Enhanced confidence framework
        factors = result["enhanced_confidence_factors"]
        calibrated = result.get("calibrated_confidence", factors.get("overall", 0.0))
        explanations = result.get("confidence_explanations", {})
        
        # Display overall confidence
        st.markdown("### Confidence Assessment")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            confidence_color = get_confidence_color_hex(calibrated)
            st.markdown(
                f"""
                <div style="
                    background-color: {confidence_color};
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    color: white;
                    font-weight: bold;
                    font-size: 24px;
                ">
                {calibrated:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            if calibrated >= 0.85:
                st.success("High confidence - Auto-accept recommended")
            elif calibrated >= 0.6:
                st.warning("Medium confidence - Verification recommended")
            else:
                st.error("Low confidence - Manual review required")
        
        # Display confidence factors visualization
        st.subheader("Confidence Factor Breakdown")
        chart = create_confidence_visualization(factors)
        st.altair_chart(chart, use_container_width=True)
        
        # Display explanations if requested
        if show_explanations and explanations:
            st.subheader("Confidence Factor Explanations") # Changed from st.expander
            for factor, explanation in explanations.items():
                if factor != "overall":  # Overall is shown separately
                    st.markdown(f"**{factor.replace('_', ' ').title()}**: {explanation}")
    
    elif "multi_factor_confidence" in result:
        # Original confidence framework
        factors = result["multi_factor_confidence"]
        calibrated = result.get("calibrated_confidence", factors.get("overall", 0.0))
        
        # Display overall confidence
        st.markdown("### Confidence Assessment")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            confidence_color = get_confidence_color_hex(calibrated)
            st.markdown(
                f"""
                <div style="
                    background-color: {confidence_color};
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    color: white;
                    font-weight: bold;
                    font-size: 24px;
                ">
                {calibrated:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            if calibrated >= 0.85:
                st.success("High confidence - Auto-accept recommended")
            elif calibrated >= 0.6:
                st.warning("Medium confidence - Verification recommended")
            else:
                st.error("Low confidence - Manual review required")
        
        # Create a DataFrame for the confidence factors
        factor_data = []
        for factor_name, factor_value in factors.items():
            if factor_name != "overall":  # Skip overall, it's displayed separately
                factor_data.append({
                    "Factor": factor_name.replace("_", " ").title(),
                    "Value": factor_value,
                    "Color": get_confidence_color_name(factor_value)
                })
        
        df = pd.DataFrame(factor_data)
        
        # Create the chart
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Value:Q', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Factor:N', sort=None),
            color=alt.Color('Color:N', scale=alt.Scale(
                domain=['Low', 'Medium', 'High'],
                range=['#FF5252', '#FFA726', '#66BB6A']
            ), legend=None),
            tooltip=['Factor', 'Value']
        ).properties(
            title='Confidence Factor Breakdown',
            width=400,
            height=200
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Display explanations
        with st.expander("Confidence Factor Explanations", expanded=False):
            st.markdown("""
            **AI Model Confidence**: Raw confidence score reported by the AI model.
            
            **Response Quality**: Assessment of how well-structured and complete the AI's response is.
            
            **Category Specificity**: How specific and definitive the category assignment is.
            
            **Reasoning Quality**: How detailed and evidence-based the reasoning is.
            
            **Document Features Match**: How well document features match the assigned category.
            """)
    else:
        # Simple confidence display if no detailed factors available
        confidence = result.get("confidence", 0.0)
        st.markdown("### Confidence Assessment")
        
        confidence_color = get_confidence_color_hex(confidence)
        st.markdown(
            f"""
            <div style="
                background-color: {confidence_color};
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 24px;
            ">
            {confidence:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if confidence >= 0.85:
            st.success("High confidence - Auto-accept recommended")
        elif confidence >= 0.6:
            st.warning("Medium confidence - Verification recommended")
        else:
            st.error("Low confidence - Manual review required")
