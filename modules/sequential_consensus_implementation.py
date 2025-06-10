import streamlit as st
import logging
import json
import requests
import re
import os
import datetime
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import uuid
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def categorize_document_with_sequential_consensus(
    file_id: str, 
    model1: str, 
    model2: str, 
    model3: str, 
    document_types_with_desc: List[Dict[str, str]],
    disagreement_threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Perform document categorization using sequential consensus approach:
    1. Model 1 performs initial categorization
    2. Model 2 performs independent categorization without seeing Model 1's results
    3. Model 2 reviews both results only after forming its own opinion
    4. Model 3 arbitrates if there's significant disagreement
    
    Args:
        file_id: Box file ID
        model1: AI model for initial categorization
        model2: AI model for independent assessment and review
        model3: AI model for arbitration (used only when needed)
        document_types_with_desc: List of document types with descriptions
        disagreement_threshold: Threshold to trigger Model 3 arbitration
        
    Returns:
        Dictionary with categorization results and consensus information
    """
    # Import directly from document_categorization_utils to avoid circular imports
    from modules.document_categorization_utils import (
        categorize_document,
        extract_document_features,
        calculate_multi_factor_confidence,
        apply_confidence_calibration
    )
    
    logger.info(f"Starting sequential consensus categorization for file {file_id}")
    
    # Step 1: Initial categorization with Model 1
    logger.info(f"Step 1: Initial categorization with {model1}")
    model1_result = categorize_document(file_id, model1, document_types_with_desc)
    model1_result["model_name"] = model1
    
    # Extract document features and calculate multi-factor confidence for Model 1
    try:
        document_features = extract_document_features(file_id)
    except AttributeError as e:
        logger.error(f"AttributeError extracting features in sequential_consensus for file {file_id}: {str(e)}. Using empty features.")
        document_features = {}
    except Exception as e:
        logger.error(f"Unexpected error extracting features in sequential_consensus for file {file_id}: {str(e)}. Using empty features.")
        document_features = {}

    valid_categories = [dtype["name"] for dtype in document_types_with_desc]
    
    model1_multi_factor_confidence = calculate_multi_factor_confidence(
        model1_result["confidence"],
        document_features,
        model1_result["document_type"],
        model1_result.get("reasoning", ""),
        valid_categories
    )
    model1_result["multi_factor_confidence"] = model1_multi_factor_confidence
    model1_result["calibrated_confidence"] = apply_confidence_calibration(
        model1_result["document_type"],
        model1_multi_factor_confidence.get("overall", model1_result["confidence"])
    )
    
    # Step 2: Completely independent assessment by Model 2 with no knowledge of Model 1's results
    logger.info(f"Step 2A: Independent assessment by {model2} (no knowledge of Model 1's results)")
    model2_independent_result = independent_categorization(file_id, model2, document_types_with_desc)
    
    # Step 2B: Only after independent assessment, Model 2 reviews both results
    logger.info(f"Step 2B: Review by {model2} (after independent assessment)")
    model2_result = review_categorization(
        file_id, 
        model2, 
        model1_result, 
        model2_independent_result, 
        document_types_with_desc
    )
    model2_result["model_name"] = model2
    
    # Calculate agreement level and confidence
    agreement_level, confidence_adjustment = calculate_agreement_confidence(model1_result, model2_result)
    
    # Determine if arbitration is needed
    needs_arbitration = False
    arbitration_reason = ""
    
    # Check for category disagreement
    if model1_result["document_type"] != model2_result["document_type"]:
        needs_arbitration = True
        arbitration_reason = "Category disagreement"
    
    # Check for significant confidence difference
    elif abs(model1_result["confidence"] - model2_result["confidence"]) > disagreement_threshold:
        needs_arbitration = True
        arbitration_reason = f"Confidence difference exceeds threshold ({disagreement_threshold})"
    
    # Step 3: Arbitration by Model 3 if needed
    model3_result = {}
    final_document_type = ""
    final_confidence = 0.0
    final_reasoning = ""
    
    if needs_arbitration:
        logger.info(f"Step 3: Arbitration needed ({arbitration_reason}). Using {model3}")
        model3_result = arbitrate_categorization(file_id, model3, model1_result, model2_result, document_types_with_desc)
        model3_result["model_name"] = model3
        
        # Use Model 3's decision as final
        final_document_type = model3_result["document_type"]
        final_confidence = model3_result["confidence"]
        final_reasoning = model3_result["reasoning"]
        agreement_level = "Arbitrated"
    else:
        logger.info("No arbitration needed, using Model 2's review as final")
        # Use Model 2's review as final (it already considered Model 1's result)
        final_document_type = model2_result["document_type"]
        final_confidence = model2_result["confidence"]
        final_reasoning = model2_result["reasoning"]
    
    # Calculate final multi-factor confidence
    final_multi_factor_confidence = calculate_multi_factor_confidence(
        final_confidence,
        document_features,
        final_document_type,
        final_reasoning,
        valid_categories
    )
    
    # Apply calibration to final confidence
    final_calibrated_confidence = apply_confidence_calibration(
        final_document_type,
        final_multi_factor_confidence.get("overall", final_confidence)
    )
    
    # Prepare sequential consensus information
    sequential_consensus = {
        "agreement_level": agreement_level,
        "confidence_adjustment": confidence_adjustment,
        "needs_arbitration": needs_arbitration,
        "arbitration_reason": arbitration_reason if needs_arbitration else ""
    }
    
    # Prepare final result
    result = {
        "document_type": final_document_type,
        "confidence": final_confidence,
        "reasoning": final_reasoning,
        "multi_factor_confidence": final_multi_factor_confidence,
        "calibrated_confidence": final_calibrated_confidence,
        "document_features": document_features,
        "sequential_consensus": sequential_consensus,
        "model1_result": model1_result,
        "model2_result": model2_result,
        "model2_independent_result": model2_independent_result  # Include the independent assessment
    }
    
    # Include Model 3 result if arbitration was used
    if needs_arbitration:
        result["model3_result"] = model3_result
    
    return result

def independent_categorization(
    file_id: str, 
    model: str, 
    document_types_with_desc: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Have a model categorize a document completely independently.
    This function is isolated from any other categorization results.
    
    Args:
        file_id: Box file ID
        model: AI model for categorization
        document_types_with_desc: List of document types with descriptions
        
    Returns:
        Dictionary with categorization results
    """
    access_token = None
    if hasattr(st.session_state.client, "_oauth"):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, "auth") and hasattr(st.session_state.client.auth, "access_token"):
        access_token = st.session_state.client.auth.access_token
    if not access_token:
        raise ValueError("Could not retrieve access token from client")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    valid_categories = [dtype["name"] for dtype in document_types_with_desc]
    category_options_text = "\n".join([f"- {dtype['name']}: {dtype['description']}" for dtype in document_types_with_desc])
    
    # Generate a unique session ID to ensure this is a completely fresh context
    session_id = str(uuid.uuid4())
    
    # Create a prompt that emphasizes independent judgment
    prompt = (
        f"Session ID: {session_id}\n\n"
        f"You are an expert document analyst with extensive experience in document classification. "
        f"Please categorize this document into one of the following categories based SOLELY on your own judgment:\n\n"
        f"Available categories:\n{category_options_text}\n\n"
        f"Important: Form your own independent opinion without any external influence. "
        f"Analyze the document thoroughly and provide your honest assessment.\n\n"
        f"Respond in the following format:\n"
        f"Category: [your selected category name]\n"
        f"Confidence: [your confidence score between 0.0 and 1.0]\n"
        f"Reasoning: [Your detailed reasoning for the categorization]"
    )

    logger.info(f"Box AI Independent Categorization Request for file {file_id} (model: {model}, session: {session_id}):\n{prompt}")

    api_url = "https://api.box.com/2.0/ai/ask"
    request_body = {
        "mode": "single_item_qa",
        "prompt": prompt,
        "items": [{"type": "file", "id": file_id}],
        "ai_agent": {"type": "ai_agent_ask", "basic_text": {"model": model, "mode": "default"}}
    }

    try:
        logger.info(f"Making Box AI independent categorization call for file {file_id} with model {model}")
        response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Box AI independent categorization response for {file_id}: {json.dumps(response_data)}")

        if "answer" in response_data and response_data["answer"]:
            original_response = response_data["answer"]
            parsed_result = parse_independent_response(original_response, valid_categories)
            parsed_result["session_id"] = session_id
            return parsed_result
        else:
            logger.warning(f"No answer field or empty answer in Box AI response for file {file_id}. Response: {response_data}")
            return {
                "document_type": "Unknown",
                "confidence": 0.5,
                "reasoning": "Model did not provide a valid response.",
                "original_response": str(response_data),
                "session_id": session_id
            }
    except Exception as e:
        logger.error(f"Error during Box AI call for file {file_id}: {str(e)}")
        return {
            "document_type": "Unknown",
            "confidence": 0.5,
            "reasoning": f"Error during categorization: {str(e)}",
            "original_response": str(e),
            "session_id": session_id
        }

def review_categorization(
    file_id: str, 
    model: str, 
    model1_result: Dict[str, Any],
    model2_independent_result: Dict[str, Any],
    document_types_with_desc: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Have Model 2 review both Model 1's results and its own independent assessment.
    
    Args:
        file_id: Box file ID
        model: AI model for review
        model1_result: Result from Model 1's categorization
        model2_independent_result: Result from Model 2's independent assessment
        document_types_with_desc: List of document types with descriptions
        
    Returns:
        Dictionary with review results
    """
    access_token = None
    if hasattr(st.session_state.client, "_oauth"):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, "auth") and hasattr(st.session_state.client.auth, "access_token"):
        access_token = st.session_state.client.auth.access_token
    if not access_token:
        raise ValueError("Could not retrieve access token from client")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    valid_categories = [dtype["name"] for dtype in document_types_with_desc]
    category_options_text = "\n".join([f"- {dtype['name']}: {dtype['description']}" for dtype in document_types_with_desc])
    
    # Extract information from results
    model1_category = model1_result["document_type"]
    model1_confidence = model1_result["confidence"]
    model1_reasoning = model1_result.get("reasoning", "No reasoning provided")
    model1_name = model1_result.get("model_name", "Model 1")
    
    independent_category = model2_independent_result["document_type"]
    independent_confidence = model2_independent_result["confidence"]
    independent_reasoning = model2_independent_result["reasoning"]
    independent_session_id = model2_independent_result.get("session_id", "unknown")
    
    # Generate a unique review session ID
    review_session_id = str(uuid.uuid4())
    
    high_confidence_threshold = 0.80

    prompt_intro = (
        f"Review Session ID: {review_session_id}\n\n"
        f"You are an expert document reviewer tasked with evaluating document categorization results. "
        f"You have two categorizations of the same document to compare:\n\n"
        
        f"1. YOUR OWN INDEPENDENT ASSESSMENT (Session ID: {independent_session_id}):\n"
        f"Category: {independent_category}\n"
        f"Confidence: {independent_confidence:.2f}\n"
        f"Your reasoning: {independent_reasoning}\n\n"
        
        f"2. ANOTHER MODEL'S ASSESSMENT ({model1_name}):\n"
        f"Category: {model1_category}\n"
        f"Confidence: {model1_confidence:.2f}\n"
        f"Their reasoning: {model1_reasoning}\n\n"
    )

    prompt_evaluation_guidance = ""
    if independent_confidence > high_confidence_threshold:
        prompt_evaluation_guidance = (
            f"IMPORTANT: Your independent assessment (Category: {independent_category}, Confidence: {independent_confidence:.2f}) was made with HIGH CONFIDENCE. \n"
            f"You should only change your category if the other model's assessment provides CLEAR AND COMPELLING evidence that your initial assessment was incorrect. \n"
            f"If you decide to change your category, you MUST explicitly state in your 'Assessment' and 'Reasoning' WHY your initial high-confidence assessment was flawed. \n"
            f"Furthermore, if you change your category, carefully consider your new confidence score. It should generally NOT be higher than your initial high confidence of {independent_confidence:.2f}, unless the other model's evidence is exceptionally strong and unequivocally justifies such an increase. You MUST explain any significant change in your confidence level in your 'Reasoning'.\n\n"
        )
    else:
        prompt_evaluation_guidance = (
            f"Please critically evaluate both assessments. Your task is to provide a final categorization based on the document's content, not simply agreeing with either previous assessment. \n"
            f"Pay attention to the reasoning and confidence of both your initial assessment and the other model's assessment. If you change your category, explain why. Also, justify your final confidence score in your reasoning.\n\n"
        )

    review_prompt = (
        f"{prompt_intro}"
        f"INSTRUCTIONS FOR YOUR REVIEW:\n"
        f"{prompt_evaluation_guidance}"
        f"After carefully considering these instructions and re-examining the document, provide your final assessment.\n\n"
        f"Available categories:\n{category_options_text}\n\n"
        f"Respond in the following format:\n"
        f"Category: [your final category selection]\n"
        f"Confidence: [your final confidence score between 0.0 and 1.0]\n"
        f"Agreement: [Agree/Partially Agree/Disagree] with the other model's categorization\n"
        f"Assessment: [Your critical assessment of the other model's categorization and reasoning, including detailed justification if you changed from a high-confidence initial assessment, explaining why it was flawed]\n"
        f"Reasoning: [Your detailed reasoning for your final categorization and confidence level, including justification for any changes from your initial assessment and confidence]"
    )

    logger.info(f"Box AI Review Request for file {file_id} (model: {model}, review session: {review_session_id}):\n{review_prompt}")

    # Add a small delay to ensure API context separation
    time.sleep(1)

    api_url = "https://api.box.com/2.0/ai/ask"
    review_request_body = {
        "mode": "single_item_qa",
        "prompt": review_prompt,
        "items": [{"type": "file", "id": file_id}],
        "ai_agent": {"type": "ai_agent_ask", "basic_text": {"model": model, "mode": "default"}}
    }

    try:
        logger.info(f"Making Box AI review call for file {file_id} with model {model}")
        review_response = requests.post(api_url, headers=headers, json=review_request_body, timeout=180)
        review_response.raise_for_status()
        review_data = review_response.json()
        logger.info(f"Box AI review response for {file_id}: {json.dumps(review_data)}")

        if "answer" in review_data and review_data["answer"]:
            original_response = review_data["answer"]
            parsed_result = parse_review_response(original_response, valid_categories, model1_result)
            
            # Add independent assessment and session info to result
            parsed_result["independent_assessment"] = model2_independent_result
            parsed_result["review_session_id"] = review_session_id
            
            # Calculate confidence adjustment factors
            confidence_adjustment_factors = {
                "agreement_bonus": 0.0,
                "disagreement_penalty": 0.0,
                "reasoning_quality": 0.0,
                "independent_consistency": 0.0
            }
            
            # Apply adjustments based on agreement level with Model 1
            if parsed_result["review_assessment"]["agreement_level"] == "Agree":
                confidence_adjustment_factors["agreement_bonus"] = 0.05
            elif parsed_result["review_assessment"]["agreement_level"] == "Partially Agree":
                confidence_adjustment_factors["agreement_bonus"] = 0.02
            else:  # Disagree
                confidence_adjustment_factors["disagreement_penalty"] = -0.05
            
            # Apply adjustments based on consistency with independent assessment
            if parsed_result["document_type"] == independent_category:
                confidence_adjustment_factors["independent_consistency"] = 0.05
            else:
                confidence_adjustment_factors["independent_consistency"] = -0.03
            
            # Adjust based on reasoning quality assessment
            reasoning_assessment = parsed_result["review_assessment"]["assessment_reasoning"].lower()
            if "excellent" in reasoning_assessment or "thorough" in reasoning_assessment or "comprehensive" in reasoning_assessment:
                confidence_adjustment_factors["reasoning_quality"] = 0.03
            elif "poor" in reasoning_assessment or "inadequate" in reasoning_assessment or "insufficient" in reasoning_assessment:
                confidence_adjustment_factors["reasoning_quality"] = -0.03
            
            parsed_result["confidence_adjustment_factors"] = confidence_adjustment_factors
            
            return parsed_result
        else:
            logger.warning(f"No answer field or empty answer in Box AI review response for file {file_id}. Response: {review_data}")
            return {
                "document_type": independent_category,  # Use independent assessment as fallback
                "confidence": independent_confidence * 0.9,  # Slightly reduce confidence due to review failure
                "reasoning": "Review failed. Using independent assessment as fallback.",
                "original_response": str(review_data),
                "independent_assessment": model2_independent_result,
                "review_session_id": review_session_id,
                "review_assessment": {
                    "agreement_level": "Unknown",
                    "assessment_reasoning": "Review failed"
                }
            }
    except Exception as e:
        logger.error(f"Error during Box AI review call for file {file_id}: {str(e)}")
        return {
            "document_type": independent_category,  # Use independent assessment as fallback
            "confidence": independent_confidence * 0.9,  # Slightly reduce confidence due to review failure
            "reasoning": f"Error during review: {str(e)}. Using independent assessment as fallback.",
            "original_response": str(e),
            "independent_assessment": model2_independent_result,
            "review_session_id": review_session_id,
            "review_assessment": {
                "agreement_level": "Unknown",
                "assessment_reasoning": f"Review error: {str(e)}"
            }
        }

def parse_independent_response(response_text: str, valid_categories: List[str]) -> Dict[str, Any]:
    """
    Parse the response from the independent assessment.
    
    Args:
        response_text: Response text from the AI model
        valid_categories: List of valid category names
        
    Returns:
        Dictionary with parsed assessment results
    """
    logger.info(f"Parsing independent assessment response: {response_text[:150]}...")
    
    # Default values
    category = "Unknown"
    confidence = 0.5
    reasoning = ""
    
    lines = response_text.strip().split("\n")
    
    # Parse each line based on expected format
    for i, line in enumerate(lines):
        if line.lower().startswith("category:"):
            extracted_category = line.split(":", 1)[1].strip()
            # Find exact or partial match
            for valid_cat in valid_categories:
                if valid_cat.lower() == extracted_category.lower():
                    category = valid_cat
                    break
            else:
                for valid_cat in valid_categories:
                    if valid_cat.lower() in extracted_category.lower() or extracted_category.lower() in valid_cat.lower():
                        category = valid_cat
                        break
        
        elif line.lower().startswith("confidence:"):
            confidence_match = re.search(r"([0-9]*\.?[0-9]+)", line)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    logger.warning(f"Could not parse confidence value: {line}")
        
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
            # Include all remaining lines as part of reasoning
            if i+1 < len(lines):
                reasoning += "\n" + "\n".join(lines[i+1:])
            break
    
    # If no reasoning found, use everything after the confidence as reasoning
    if not reasoning:
        for i, line in enumerate(lines):
            if line.lower().startswith("confidence:"):
                if i+1 < len(lines):
                    reasoning = "\n".join(lines[i+1:])
                break
    
    # If still no reasoning, use the original response
    if not reasoning:
        reasoning = response_text
    
    logger.info(f"Parsed independent assessment - Category: {category}, Confidence: {confidence:.2f}")
    
    return {
        "document_type": category,
        "confidence": confidence,
        "reasoning": reasoning,
        "original_response": response_text
    }

def parse_review_response(response_text: str, valid_categories: List[str], initial_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the response from the review model.
    
    Args:
        response_text: Response text from the AI model
        valid_categories: List of valid category names
        initial_result: Result from the initial categorization
        
    Returns:
        Dictionary with parsed review results
    """
    logger.info(f"Parsing review response: {response_text[:150]}...")
    
    # Default to initial result values
    category = initial_result["document_type"]
    confidence = initial_result["confidence"]
    agreement_level = "Unknown"
    assessment_reasoning = ""
    reasoning = ""
    
    lines = response_text.strip().split("\n")
    
    # Parse each line based on expected format
    for i, line in enumerate(lines):
        if line.lower().startswith("category:"):
            extracted_category = line.split(":", 1)[1].strip()
            # Find exact or partial match
            for valid_cat in valid_categories:
                if valid_cat.lower() == extracted_category.lower():
                    category = valid_cat
                    break
            else:
                for valid_cat in valid_categories:
                    if valid_cat.lower() in extracted_category.lower() or extracted_category.lower() in valid_cat.lower():
                        category = valid_cat
                        break
        
        elif line.lower().startswith("confidence:"):
            confidence_match = re.search(r"([0-9]*\.?[0-9]+)", line)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    logger.warning(f"Could not parse confidence value: {line}")
        
        elif line.lower().startswith("agreement:"):
            agreement_text = line.split(":", 1)[1].strip().lower()
            if "agree" in agreement_text and "partially" not in agreement_text and "dis" not in agreement_text:
                agreement_level = "Agree"
            elif "partially" in agreement_text or "somewhat" in agreement_text:
                agreement_level = "Partially Agree"
            elif "disagree" in agreement_text or "dis agree" in agreement_text:
                agreement_level = "Disagree"
            else:
                agreement_level = "Unknown"
        
        elif line.lower().startswith("assessment:"):
            assessment_reasoning = line.split(":", 1)[1].strip()
            # Check if assessment continues on next lines (until we hit Reasoning:)
            for j in range(i+1, len(lines)):
                if lines[j].lower().startswith("reasoning:"):
                    break
                assessment_reasoning += " " + lines[j]
        
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
            # Include all remaining lines as part of reasoning
            if i+1 < len(lines):
                reasoning += "\n" + "\n".join(lines[i+1:])
            break
    
    # If no reasoning found, use everything after the Assessment as reasoning
    if not reasoning:
        for i, line in enumerate(lines):
            if line.lower().startswith("assessment:"):
                if i+1 < len(lines):
                    reasoning = "\n".join(lines[i+1:])
                break
    
    # If still no reasoning, use the original response
    if not reasoning:
        reasoning = response_text
    
    # If no assessment reasoning found, create a default one
    if not assessment_reasoning:
        if agreement_level == "Agree":
            assessment_reasoning = "The original categorization appears correct."
        elif agreement_level == "Partially Agree":
            assessment_reasoning = "The original categorization is partially correct but could be improved."
        elif agreement_level == "Disagree":
            assessment_reasoning = "The original categorization appears incorrect."
        else:
            assessment_reasoning = "Unable to determine agreement with original categorization."
    
    logger.info(f"Parsed review - Category: {category}, Confidence: {confidence:.2f}, Agreement: {agreement_level}")
    
    return {
        "document_type": category,
        "confidence": confidence,
        "reasoning": reasoning,
        "original_response": response_text,
        "review_assessment": {
            "agreement_level": agreement_level,
            "assessment_reasoning": assessment_reasoning
        }
    }

def arbitrate_categorization(
    file_id: str, 
    model: str, 
    model1_result: Dict[str, Any], 
    model2_result: Dict[str, Any],
    document_types_with_desc: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Have a third model arbitrate between conflicting categorization results.
    
    Args:
        file_id: Box file ID
        model: AI model for arbitration
        model1_result: Result from the initial categorization
        model2_result: Result from the review
        document_types_with_desc: List of document types with descriptions
        
    Returns:
        Dictionary with arbitration results
    """
    access_token = None
    if hasattr(st.session_state.client, "_oauth"):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, "auth") and hasattr(st.session_state.client.auth, "access_token"):
        access_token = st.session_state.client.auth.access_token
    if not access_token:
        raise ValueError("Could not retrieve access token from client")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    valid_categories = [dtype["name"] for dtype in document_types_with_desc]
    category_options_text = "\n".join([f"- {dtype['name']}: {dtype['description']}" for dtype in document_types_with_desc])
    
    model1_category = model1_result["document_type"]
    model1_confidence = model1_result["confidence"]
    model1_reasoning = model1_result.get("reasoning", "No reasoning provided")
    model1_name = model1_result.get("model_name", "Model 1")
    
    model2_category = model2_result["document_type"]
    model2_confidence = model2_result["confidence"]
    model2_reasoning = model2_result.get("reasoning", "No reasoning provided")
    model2_name = model2_result.get("model_name", "Model 2")
    
    # Include independent assessment if available
    independent_assessment_text = ""
    if "independent_assessment" in model2_result:
        independent_category = model2_result["independent_assessment"]["document_type"]
        independent_confidence = model2_result["independent_assessment"]["confidence"]
        independent_reasoning = model2_result["independent_assessment"]["reasoning"]
        
        independent_assessment_text = (
            f"MODEL 2 INDEPENDENT ASSESSMENT (before seeing Model 1's results):\n"
            f"Category: {independent_category}\n"
            f"Confidence: {independent_confidence:.2f}\n"
            f"Reasoning: {independent_reasoning}\n\n"
        )
    
    # Generate a unique arbitration session ID
    arbitration_session_id = str(uuid.uuid4())
    
    prompt = (
        f"Arbitration Session ID: {arbitration_session_id}\n\n"
        f"You are an expert document arbitrator. Two AI models have categorized the same document with different results:\n\n"
        
        f"MODEL 1 ({model1_name}) CATEGORIZATION:\n"
        f"Category: {model1_category}\n"
        f"Confidence: {model1_confidence:.2f}\n"
        f"Reasoning: {model1_reasoning}\n\n"
        
        f"MODEL 2 ({model2_name}) CATEGORIZATION:\n"
        f"{independent_assessment_text}"
        f"Category: {model2_category}\n"
        f"Confidence: {model2_confidence:.2f}\n"
        f"Reasoning: {model2_reasoning}\n\n"
        
        f"Please examine the document yourself and arbitrate between these conflicting categorizations. "
        f"Your task is to provide an independent, unbiased assessment based on the document's content, "
        f"not simply choosing one of the previous assessments.\n\n"
        
        f"You should:\n"
        f"1. Determine which model's categorization is more accurate (or provide your own if both are incorrect)\n"
        f"2. Provide your own confidence score\n"
        f"3. Assess the quality of each model's reasoning\n"
        f"4. Provide your own detailed reasoning\n\n"
        
        f"Available categories:\n{category_options_text}\n\n"
        
        f"Respond in the following format:\n"
        f"Category: [your selected category name]\n"
        f"Confidence: [your confidence score between 0.0 and 1.0]\n"
        f"Model 1 Assessment: [Your assessment of Model 1's categorization]\n"
        f"Model 2 Assessment: [Your assessment of Model 2's categorization]\n"
        f"Arbitration: [Your explanation of which model is more accurate and why]\n"
        f"Reasoning: [Your detailed reasoning for the final categorization]"
    )

    logger.info(f"Box AI Arbitration Request for file {file_id} (model: {model}, arbitration session: {arbitration_session_id}):\n{prompt}")

    # Add a small delay to ensure API context separation
    time.sleep(1)

    api_url = "https://api.box.com/2.0/ai/ask"
    request_body = {
        "mode": "single_item_qa",
        "prompt": prompt,
        "items": [{"type": "file", "id": file_id}],
        "ai_agent": {"type": "ai_agent_ask", "basic_text": {"model": model, "mode": "default"}}
    }

    try:
        logger.info(f"Making Box AI arbitration call for file {file_id} with model {model}")
        response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Box AI arbitration response for {file_id}: {json.dumps(response_data)}")

        if "answer" in response_data and response_data["answer"]:
            original_response = response_data["answer"]
            parsed_result = parse_arbitration_response(original_response, valid_categories, model1_result, model2_result)
            parsed_result["arbitration_session_id"] = arbitration_session_id
            
            # Calculate confidence factors based on arbitration
            confidence_factors = {
                "model_agreement": 0.0,
                "reasoning_quality": 0.0,
                "arbitration_confidence": 0.0
            }
            
            # Check if arbitration agrees with either model
            if parsed_result["document_type"] == model1_result["document_type"]:
                confidence_factors["model_agreement"] = 0.05
            elif parsed_result["document_type"] == model2_result["document_type"]:
                confidence_factors["model_agreement"] = 0.05
            
            # Assess reasoning quality based on arbitration assessment
            arbitration_reasoning = parsed_result["arbitration_assessment"]["arbitration_reasoning"].lower()
            if "clear" in arbitration_reasoning or "strong" in arbitration_reasoning or "definitive" in arbitration_reasoning:
                confidence_factors["reasoning_quality"] = 0.05
            elif "uncertain" in arbitration_reasoning or "ambiguous" in arbitration_reasoning:
                confidence_factors["reasoning_quality"] = -0.05
            
            # Arbitration confidence factor
            confidence_factors["arbitration_confidence"] = parsed_result["confidence"] * 0.1
            
            parsed_result["confidence_factors"] = confidence_factors
            
            return parsed_result
        else:
            logger.warning(f"No answer field or empty answer in Box AI arbitration response for file {file_id}. Response: {response_data}")
            # Fall back to the model with higher confidence
            if model1_result["confidence"] >= model2_result["confidence"]:
                fallback_result = model1_result
                fallback_message = "Arbitration failed. Using Model 1's result (higher confidence)."
            else:
                fallback_result = model2_result
                fallback_message = "Arbitration failed. Using Model 2's result (higher confidence)."
                
            return {
                "document_type": fallback_result["document_type"],
                "confidence": fallback_result["confidence"] * 0.9,  # Slightly reduce confidence due to arbitration failure
                "reasoning": fallback_message + "\n\nOriginal reasoning: " + fallback_result.get("reasoning", ""),
                "original_response": str(response_data),
                "arbitration_session_id": arbitration_session_id,
                "arbitration_assessment": {
                    "model1_assessment": "Arbitration failed",
                    "model2_assessment": "Arbitration failed",
                    "arbitration_reasoning": fallback_message
                }
            }
    except Exception as e:
        logger.error(f"Error during Box AI arbitration call for file {file_id}: {str(e)}")
        # Fall back to the model with higher confidence
        if model1_result["confidence"] >= model2_result["confidence"]:
            fallback_result = model1_result
            fallback_message = f"Arbitration error: {str(e)}. Using Model 1's result (higher confidence)."
        else:
            fallback_result = model2_result
            fallback_message = f"Arbitration error: {str(e)}. Using Model 2's result (higher confidence)."
            
        return {
            "document_type": fallback_result["document_type"],
            "confidence": fallback_result["confidence"] * 0.9,  # Slightly reduce confidence due to arbitration failure
            "reasoning": fallback_message + "\n\nOriginal reasoning: " + fallback_result.get("reasoning", ""),
            "original_response": str(e),
            "arbitration_session_id": arbitration_session_id,
            "arbitration_assessment": {
                "model1_assessment": "Arbitration error",
                "model2_assessment": "Arbitration error",
                "arbitration_reasoning": fallback_message
            }
        }

def parse_arbitration_response(
    response_text: str, 
    valid_categories: List[str], 
    model1_result: Dict[str, Any], 
    model2_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Parse the response from the arbitration model.
    
    Args:
        response_text: Response text from the AI model
        valid_categories: List of valid category names
        model1_result: Result from the initial categorization
        model2_result: Result from the review
        
    Returns:
        Dictionary with parsed arbitration results
    """
    logger.info(f"Parsing arbitration response: {response_text[:150]}...")
    
    # Default to model with higher confidence
    if model1_result["confidence"] >= model2_result["confidence"]:
        category = model1_result["document_type"]
        confidence = model1_result["confidence"]
    else:
        category = model2_result["document_type"]
        confidence = model2_result["confidence"]
    
    model1_assessment = ""
    model2_assessment = ""
    arbitration_reasoning = ""
    reasoning = ""
    
    lines = response_text.strip().split("\n")
    
    # Parse each line based on expected format
    for i, line in enumerate(lines):
        if line.lower().startswith("category:"):
            extracted_category = line.split(":", 1)[1].strip()
            # Find exact or partial match
            for valid_cat in valid_categories:
                if valid_cat.lower() == extracted_category.lower():
                    category = valid_cat
                    break
            else:
                for valid_cat in valid_categories:
                    if valid_cat.lower() in extracted_category.lower() or extracted_category.lower() in valid_cat.lower():
                        category = valid_cat
                        break
        
        elif line.lower().startswith("confidence:"):
            confidence_match = re.search(r"([0-9]*\.?[0-9]+)", line)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    logger.warning(f"Could not parse confidence value: {line}")
        
        elif line.lower().startswith("model 1 assessment:"):
            model1_assessment = line.split(":", 1)[1].strip()
            # Check if assessment continues on next lines (until we hit another section)
            for j in range(i+1, len(lines)):
                if lines[j].lower().startswith("model 2 assessment:") or \
                   lines[j].lower().startswith("arbitration:") or \
                   lines[j].lower().startswith("reasoning:"):
                    break
                model1_assessment += " " + lines[j]
        
        elif line.lower().startswith("model 2 assessment:"):
            model2_assessment = line.split(":", 1)[1].strip()
            # Check if assessment continues on next lines (until we hit another section)
            for j in range(i+1, len(lines)):
                if lines[j].lower().startswith("arbitration:") or \
                   lines[j].lower().startswith("reasoning:"):
                    break
                model2_assessment += " " + lines[j]
        
        elif line.lower().startswith("arbitration:"):
            arbitration_reasoning = line.split(":", 1)[1].strip()
            # Check if arbitration continues on next lines (until we hit Reasoning:)
            for j in range(i+1, len(lines)):
                if lines[j].lower().startswith("reasoning:"):
                    break
                arbitration_reasoning += " " + lines[j]
        
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
            # Include all remaining lines as part of reasoning
            if i+1 < len(lines):
                reasoning += "\n" + "\n".join(lines[i+1:])
            break
    
    # If no reasoning found, use everything after the Arbitration as reasoning
    if not reasoning:
        for i, line in enumerate(lines):
            if line.lower().startswith("arbitration:"):
                if i+1 < len(lines):
                    reasoning = "\n".join(lines[i+1:])
                break
    
    # If still no reasoning, use the original response
    if not reasoning:
        reasoning = response_text
    
    # If no model assessments or arbitration reasoning found, create default ones
    if not model1_assessment:
        model1_assessment = f"Model 1 categorized as {model1_result['document_type']} with confidence {model1_result['confidence']:.2f}."
    
    if not model2_assessment:
        model2_assessment = f"Model 2 categorized as {model2_result['document_type']} with confidence {model2_result['confidence']:.2f}."
    
    if not arbitration_reasoning:
        if category == model1_result["document_type"]:
            arbitration_reasoning = "After review, Model 1's categorization appears more accurate."
        elif category == model2_result["document_type"]:
            arbitration_reasoning = "After review, Model 2's categorization appears more accurate."
        else:
            arbitration_reasoning = "After review, neither model's categorization appears fully accurate. A different category has been selected."
    
    logger.info(f"Parsed arbitration - Category: {category}, Confidence: {confidence:.2f}")
    
    return {
        "document_type": category,
        "confidence": confidence,
        "reasoning": reasoning,
        "original_response": response_text,
        "arbitration_assessment": {
            "model1_assessment": model1_assessment,
            "model2_assessment": model2_assessment,
            "arbitration_reasoning": arbitration_reasoning
        }
    }

def calculate_agreement_confidence(model1_result: Dict[str, Any], model2_result: Dict[str, Any]) -> Tuple[str, float]:
    """
    Calculate agreement level and confidence adjustment based on model results.
    
    Args:
        model1_result: Result from the initial categorization
        model2_result: Result from the review
        
    Returns:
        Tuple of (agreement_level, confidence_adjustment)
    """
    # Check if categories match
    categories_match = model1_result["document_type"] == model2_result["document_type"]
    
    # Check confidence difference
    confidence_diff = abs(model1_result["confidence"] - model2_result["confidence"])
    
    # Check review assessment
    review_agreement = model2_result.get("review_assessment", {}).get("agreement_level", "Unknown")
    
    # Check consistency with independent assessment
    independent_consistency = False
    if "independent_assessment" in model2_result:
        independent_category = model2_result["independent_assessment"]["document_type"]
        independent_consistency = model2_result["document_type"] == independent_category
    
    # Determine agreement level
    if categories_match and review_agreement == "Agree" and confidence_diff < 0.1:
        agreement_level = "Full Agreement"
        confidence_adjustment = 0.05  # Boost confidence for full agreement
    elif categories_match and (review_agreement == "Partially Agree" or confidence_diff < 0.2):
        agreement_level = "Partial Agreement"
        confidence_adjustment = 0.02  # Small boost for partial agreement
    else:
        agreement_level = "Disagreement"
        confidence_adjustment = -0.05  # Penalty for disagreement
    
    # Additional adjustment for independent consistency
    if independent_consistency:
        confidence_adjustment += 0.03
    
    return agreement_level, confidence_adjustment
