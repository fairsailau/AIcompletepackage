# Utility functions for document categorization
import streamlit as st
import logging
import json
import requests
import re
import os
import datetime as dt_module # Alias to avoid conflict with datetime class
from datetime import datetime # Specific import for datetime class
import pandas as pd
import altair as alt
from typing import Dict, Any, List, Optional, Tuple

try:
    from dateutil import parser as dateutil_parser
    dateutil_parser_available = True
except ImportError:
    dateutil_parser_available = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Core Categorization Logic ---

def categorize_document(file_id: str, model: str, document_types_with_desc: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Categorize a single document using the specified AI model.
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
    category_options_text = "\n".join([f"- {dtype["name"]}: {dtype["description"]}" for dtype in document_types_with_desc])

    prompt = (
        f"Please analyze this document and categorize it into one of the following categories:\n"
        f"{category_options_text}\n\n"
        f"Respond ONLY in the following format (exactly two lines, followed by reasoning on a new line):\n"
        f"Category: [selected category name]\n"
        f"Confidence: [confidence score between 0.0 and 1.0]\n"
        f"Reasoning: [Your detailed reasoning for the categorization]"
    )

    logger.info(f"Box AI Request Prompt for file {file_id} (model: {model}):\n{prompt}")

    api_url = "https://api.box.com/2.0/ai/ask"
    request_body = {
        "mode": "single_item_qa",
        "prompt": prompt,
        "items": [{"type": "file", "id": file_id}],
        "ai_agent": {"type": "ai_agent_ask", "basic_text": {"model": model, "mode": "default"}}
    }

    try:
        logger.info(f"Making Box AI call for file {file_id} with model {model}")
        response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Box AI response for {file_id}: {json.dumps(response_data)}")

        if "answer" in response_data and response_data["answer"]:
            original_response = response_data["answer"]
            parsed_result = parse_categorization_response(original_response, valid_categories)
            return {
                "document_type": parsed_result["category"],
                "confidence": parsed_result["confidence"],
                "reasoning": parsed_result["reasoning"],
                "original_response": original_response
            }
        else:
            logger.warning(f"No answer field or empty answer in Box AI response for file {file_id}. Response: {response_data}")
            return {"document_type": "Other", "confidence": 0.0, "reasoning": "AI model did not provide a valid response.", "original_response": str(response_data)}
    except Exception as e:
        logger.error(f"Error during Box AI call for file {file_id}: {str(e)}")
        return {"document_type": "Other", "confidence": 0.0, "reasoning": f"Error during categorization: {str(e)}", "original_response": str(e)}

def categorize_document_detailed(file_id: str, model: str, initial_category: str, document_types_with_desc: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Perform a more detailed categorization analysis, focusing on the initial category.
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
    category_options_text = "\n".join([f"- {dtype["name"]}: {dtype["description"]}" for dtype in document_types_with_desc])

    prompt = (
        f"An initial analysis suggested this document might be categorized as 	\"{initial_category}\".\n"
        f"Please perform a more detailed analysis. Confirm if 	\"{initial_category}\" is the best fit, or suggest a more appropriate category from the list below. Provide strong reasoning.\n\n"
        f"Categories:\n{category_options_text}\n\n"
        f"Respond ONLY in the following format (exactly two lines, followed by reasoning on a new line):\n"
        f"Category: [selected category name]\n"
        f"Confidence: [confidence score between 0.0 and 1.0]\n"
        f"Reasoning: [Your detailed reasoning for the categorization, explaining why it fits or why the initial category was incorrect]"
    )

    logger.info(f"Box AI Detailed Request Prompt for file {file_id} (model: {model}):\n{prompt}")

    api_url = "https://api.box.com/2.0/ai/ask"
    request_body = {
        "mode": "single_item_qa",
        "prompt": prompt,
        "items": [{"type": "file", "id": file_id}],
        "ai_agent": {"type": "ai_agent_ask", "basic_text": {"model": model, "mode": "default"}}
    }

    try:
        logger.info(f"Making Box AI detailed call for file {file_id} with model {model}")
        response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Box AI detailed response for {file_id}: {json.dumps(response_data)}")

        if "answer" in response_data and response_data["answer"]:
            original_response = response_data["answer"]
            parsed_result = parse_categorization_response(original_response, valid_categories)
            return {
                "document_type": parsed_result["category"],
                "confidence": parsed_result["confidence"],
                "reasoning": parsed_result["reasoning"],
                "original_response": original_response
            }
        else:
            logger.warning(f"No answer field or empty answer in Box AI detailed response for file {file_id}. Response: {response_data}")
            return {"document_type": initial_category, "confidence": 0.1, "reasoning": "Detailed analysis failed to provide a valid response.", "original_response": str(response_data)}
    except Exception as e:
        logger.error(f"Error during Box AI detailed call for file {file_id}: {str(e)}")
        return {"document_type": initial_category, "confidence": 0.1, "reasoning": f"Error during detailed categorization: {str(e)}", "original_response": str(e)}

def parse_categorization_response(response_text: str, valid_categories: List[str]) -> Dict[str, Any]:
    """
    Parse the response from the AI model to extract category, confidence, and reasoning.
    """
    logger.info(f"Parsing response: {response_text[:150]}...")
    category = "Other"
    confidence = 0.0
    reasoning = ""

    lines = response_text.strip().split("\n")
    category_found = False
    confidence_found = False

    if len(lines) >= 2:
        # Parse Category
        category_match = re.match(r"Category:\s*(.*)", lines[0], re.IGNORECASE)
        if category_match:
            extracted_category = category_match.group(1).strip()
            # Find exact or partial match
            found_match = False
            for valid_cat in valid_categories:
                if valid_cat.lower() == extracted_category.lower():
                    category = valid_cat
                    found_match = True
                    break
            if not found_match:
                for valid_cat in valid_categories:
                    if valid_cat.lower() in extracted_category.lower() or extracted_category.lower() in valid_cat.lower():
                        category = valid_cat
                        found_match = True
                        break
            if not found_match and extracted_category.lower() == "other":
                category = "Other"
            category_found = True

        # Parse Confidence
        confidence_match = re.match(r"Confidence:\s*([0-9]*\.?[0-9]+)", lines[1], re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
                confidence_found = True
            except ValueError:
                logger.warning(f"Could not parse confidence value: {confidence_match.group(1)}")

        # Parse Reasoning
        if len(lines) > 2:
            reasoning_match = re.match(r"Reasoning:\s*(.*)", lines[2], re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                if len(lines) > 3:
                    reasoning += "\n" + "\n".join(lines[3:])
            else:
                # If the third line doesn't start with Reasoning:, assume it's part of the reasoning
                reasoning = "\n".join(lines[2:])
        elif not reasoning: # If no reasoning line, use the original response minus first two lines
             reasoning = response_text

    if not category_found or not confidence_found:
        logger.warning(f"Could not parse category or confidence reliably from response: {response_text}")
        # Fallback: Try to find category and confidence anywhere in the text
        if not category_found:
            for valid_cat in valid_categories:
                if valid_cat.lower() in response_text.lower():
                    category = valid_cat
                    break
        if not confidence_found:
            confidence_search = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", response_text, re.IGNORECASE)
            if confidence_search:
                try:
                    confidence = float(confidence_search.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
        if not reasoning:
             reasoning = f"Could not parse response reliably. Original: {response_text}"

    logger.info(f"Parsed - Category: {category}, Confidence: {confidence:.2f}")
    return {"category": category, "confidence": confidence, "reasoning": reasoning}

def combine_categorization_results(results: List[Dict[str, Any]], valid_categories: List[str], model_names: List[str]) -> Dict[str, Any]:
    """
    Combine results from multiple models using weighted voting based on confidence.
    """
    if not results:
        return {"document_type": "Other", "confidence": 0.0, "reasoning": "No results to combine.", "consensus_results": []}

    category_votes = {cat: 0.0 for cat in valid_categories}
    total_confidence = 0.0
    reasoning_parts = []

    for i, result in enumerate(results):
        model_name = model_names[i] if i < len(model_names) else f"Model_{i+1}"
        category = result.get("document_type", "Other")
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")

        if category in category_votes:
            # Weight vote by confidence
            category_votes[category] += confidence
        else:
            # Handle case where model returns an unexpected category
            category_votes["Other"] += confidence * 0.5 # Penalize unexpected categories

        total_confidence += confidence
        reasoning_parts.append(f"Model vote: {category} (confidence: {confidence:.2f}) Reasoning: {reasoning}")

    # Determine winning category
    winning_category = max(category_votes, key=category_votes.get)

    # Calculate final confidence: Average confidence of models that voted for the winning category
    winning_confidences = [r["confidence"] for r in results if r.get("document_type") == winning_category]
    final_confidence = sum(winning_confidences) / len(winning_confidences) if winning_confidences else 0.0

    # Adjust confidence based on agreement level
    num_models = len(results)
    agreement_ratio = len(winning_confidences) / num_models if num_models > 0 else 0
    if agreement_ratio == 1.0: # Full agreement
        final_confidence = min(1.0, final_confidence + 0.05)
    elif agreement_ratio < 0.5: # Low agreement
        final_confidence *= 0.9

    combined_reasoning = f"Consensus from models: {', '.join(model_names)}\n\nCombined result from multiple models:\nFinal category: {winning_category} (confidence: {final_confidence:.2f})\n\nIndividual model results:\n" + "\n\n".join(reasoning_parts)

    return {
        "document_type": winning_category,
        "confidence": final_confidence,
        "reasoning": combined_reasoning,
        "consensus_results": results # Keep individual results for detailed view
    }

# --- Confidence Calculation and Features ---

def extract_document_features(file_id: str) -> Dict[str, Any]:
    """
    Extract basic document features using Box API (placeholder).
    In a real implementation, this might involve more sophisticated analysis.
    """
    try:
        file_info = st.session_state.client.file(file_id).get(fields=["size", "name", "created_at", "modified_at", "parent"])

        created_date_str = "N/A"
        raw_created_at = file_info.created_at
        if isinstance(raw_created_at, str):
            try:
                dt_created = None
                if dateutil_parser_available:
                    try:
                        dt_created = dateutil_parser.isoparse(raw_created_at)
                    except Exception as de: # Broad exception for dateutil if it fails
                        logger.warning(f"dateutil.parser failed for created_at '{raw_created_at}': {de}. Falling back.")
                        # Fall through to fromisoformat logic by not setting dt_created

                if dt_created is None: # Fallback or if dateutil was not available/failed
                    processed_created_at_str = raw_created_at
                    if processed_created_at_str.endswith('Z'):
                        processed_created_at_str = processed_created_at_str[:-1] + '+00:00'
                    dt_created = datetime.fromisoformat(processed_created_at_str)

                created_date_str = dt_created.strftime("%Y-%m-%d")
            except ValueError as ve:
                logger.warning(f"Could not parse created_at string '{raw_created_at}' for file {file_id}: {ve}")
                created_date_str = raw_created_at # Fallback to original string
            except Exception as ex:
                logger.warning(f"Generic error processing created_at string '{raw_created_at}' for file {file_id}: {ex}")
                created_date_str = raw_created_at # Fallback to original string
        elif isinstance(raw_created_at, dt_module.datetime) or isinstance(raw_created_at, datetime): # Check for both aliased and direct datetime
            try:
                created_date_str = raw_created_at.strftime("%Y-%m-%d")
            except Exception as ex:
                logger.warning(f"Error formatting datetime object created_at for file {file_id}: {ex}")
                # created_date_str remains "N/A"

        modified_date_str = "N/A"
        raw_modified_at = file_info.modified_at
        if isinstance(raw_modified_at, str):
            try:
                dt_modified = None
                if dateutil_parser_available:
                    try:
                        dt_modified = dateutil_parser.isoparse(raw_modified_at)
                    except Exception as de:
                        logger.warning(f"dateutil.parser failed for modified_at '{raw_modified_at}': {de}. Falling back.")

                if dt_modified is None: # Fallback or if dateutil was not available/failed
                    processed_modified_at_str = raw_modified_at
                    if processed_modified_at_str.endswith('Z'):
                        processed_modified_at_str = processed_modified_at_str[:-1] + '+00:00'
                    dt_modified = datetime.fromisoformat(processed_modified_at_str)

                modified_date_str = dt_modified.strftime("%Y-%m-%d")
            except ValueError as ve:
                logger.warning(f"Could not parse modified_at string '{raw_modified_at}' for file {file_id}: {ve}")
                modified_date_str = raw_modified_at # Fallback to original string
            except Exception as ex:
                logger.warning(f"Generic error processing modified_at string '{raw_modified_at}' for file {file_id}: {ex}")
                modified_date_str = raw_modified_at # Fallback to original string
        elif isinstance(raw_modified_at, dt_module.datetime) or isinstance(raw_modified_at, datetime): # Check for both aliased and direct datetime
            try:
                modified_date_str = raw_modified_at.strftime("%Y-%m-%d")
            except Exception as ex:
                logger.warning(f"Error formatting datetime object modified_at for file {file_id}: {ex}")

        return {
            "file_size_kb": round(file_info.size / 1024, 2),
            "file_extension": os.path.splitext(file_info.name)[1].lower(),
            "created_date": created_date_str,
            "modified_date": modified_date_str
        }
    except Exception as e:
        logger.error(f"Error extracting features for file {file_id}: {str(e)}")
        # Return a dict with default values for all expected keys in case of error
        return {
            "file_size_kb": 0,
            "file_extension": "",
            "created_date": "N/A",
            "modified_date": "N/A"
        }

def calculate_multi_factor_confidence(
    ai_reported_confidence: float,
    document_features: Dict[str, Any],
    assigned_category: str,
    reasoning: str,
    valid_categories: List[str]
) -> Dict[str, float]:
    """
    Calculate a multi-factor confidence score based on various signals.
    """
    factors = {
        "ai_reported": ai_reported_confidence,
        "response_quality": 0.0,
        "category_specificity": 0.0,
        "reasoning_quality": 0.0,
        "document_features_match": 0.0,
        "overall": 0.0
    }

    # 1. Response Quality (Is the reasoning present and reasonably long?)
    if reasoning and len(reasoning) > 30:
        factors["response_quality"] = 0.8
    elif reasoning:
        factors["response_quality"] = 0.5
    else:
        factors["response_quality"] = 0.2

    # 2. Category Specificity (Is the assigned category specific or 'Other'?)
    if assigned_category != "Other":
        factors["category_specificity"] = 0.9
    else:
        factors["category_specificity"] = 0.3

    # 3. Reasoning Quality (Does reasoning mention keywords related to the category?)
    # Simple keyword check - could be much more sophisticated
    reasoning_lower = reasoning.lower()
    category_keywords = assigned_category.lower().split()
    keywords_found = sum(1 for keyword in category_keywords if keyword in reasoning_lower)
    if keywords_found >= 1:
        factors["reasoning_quality"] = 0.8
    elif len(reasoning) > 50: # Longer reasoning might be better even without keywords
        factors["reasoning_quality"] = 0.6
    else:
        factors["reasoning_quality"] = 0.4

    # 4. Document Features Match (Simple rules based on extension/size - placeholder)
    ext = document_features.get("file_extension", "")
    size_kb = document_features.get("file_size_kb", 0)
    feature_score = 0.5 # Default
    if assigned_category == "Invoices" and ext in [".pdf", ".docx"] and size_kb < 1024:
        feature_score = 0.7
    elif assigned_category == "Sales Contract" and ext in [".pdf", ".docx"] and size_kb > 50:
        feature_score = 0.7
    elif assigned_category == "Financial Report" and ext in [".xlsx", ".pdf", ".csv"]:
        feature_score = 0.6
    factors["document_features_match"] = feature_score

    # 5. Overall Confidence (Weighted average)
    weights = {
        "ai_reported": 0.4,
        "response_quality": 0.1,
        "category_specificity": 0.2,
        "reasoning_quality": 0.15,
        "document_features_match": 0.15
    }
    overall_score = sum(factors[key] * weights[key] for key in weights)
    factors["overall"] = min(1.0, max(0.0, overall_score))

    return factors

def apply_confidence_calibration(category: str, confidence: float) -> float:
    """
    Apply simple calibration rules based on category (placeholder).
    """
    # Example: Be slightly more skeptical about 'Other' category
    if category == "Other":
        return confidence * 0.9
    # Example: Boost confidence slightly for very common types if high
    elif category in ["Invoices", "Tax"] and confidence > 0.8:
        return min(1.0, confidence + 0.05)
    return confidence

def apply_confidence_thresholds(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Apply status labels based on confidence thresholds.
    """
    thresholds = st.session_state.get("confidence_thresholds", {
        "auto_accept": 0.85,
        "verification": 0.6,
        "rejection": 0.4
    })
    
    for file_id, result in results.items():
        confidence = result.get("calibrated_confidence", result.get("multi_factor_confidence", {}).get("overall", result.get("confidence", 0.0)))
        if confidence >= thresholds["auto_accept"]:
            result["status"] = "Auto-Accepted"
        elif confidence >= thresholds["verification"]:
            result["status"] = "Needs Verification"
        elif confidence >= thresholds["rejection"]:
            result["status"] = "Likely Incorrect"
        else:
            result["status"] = "Low Confidence / Reject"
    return results
