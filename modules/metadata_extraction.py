import streamlit as st
import logging
import json
import requests
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This function was previously named metadata_extraction
# Renaming it to get_extraction_functions to match the import in processing.py
def get_extraction_functions() -> Dict[str, Any]:
    """
    Returns a dictionary of available metadata extraction functions.
    
    Returns:
        dict: Dictionary mapping extraction method names to function objects.
    """

    def extract_structured_metadata(client: Any, file_id: str, fields: Optional[List[Dict[str, Any]]] = None, metadata_template: Optional[Dict[str, Any]] = None, ai_model: str = 'azure__openai__gpt_4o_mini') -> Dict[str, Any]:
        """
        Extract structured metadata from a file using Box AI API
        
        Args:
            client (Any): The Box API client.
            file_id (str): Box file ID
            fields (list, optional): List of field definitions for extraction
            metadata_template (dict, optional): Metadata template definition
            ai_model (str): AI model to use for extraction
            
        Returns:
            dict: Extracted metadata with confidence scores
        """
        logger.info(f"Starting structured metadata extraction for file_id: {file_id} using AI model: {ai_model}")
        try:
            # client = st.session_state.client # Client is now passed as an argument
            access_token = None
            if hasattr(client, '_oauth'):
                access_token = client._oauth.access_token
            elif hasattr(client, 'auth') and hasattr(client.auth, 'access_token'):
                access_token = client.auth.access_token
            if not access_token:
                logger.error("Could not retrieve access token from client.")
                raise ValueError('Could not retrieve access token from client')

            headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
            ai_agent = {
                'type': 'ai_agent_extract_structured', # Verified as per requirement
                'long_text': {
                    'model': ai_model,
                    'mode': 'default',
                    'system_message': 'You are an AI assistant specialized in extracting metadata from documents based on provided field definitions. For each field, analyze the document content and extract the corresponding value. CRITICALLY IMPORTANT: Respond for EACH field with a JSON object containing two keys: 1. "value": The extracted metadata value as a string. 2. "confidence": Your confidence level for this specific extraction, chosen from ONLY these three options: "High", "Medium", or "Low". Base your confidence on how certain you are about the extracted value given the document content and field definition. Example Response for a field: {"value": "INV-12345", "confidence": "High"}'
                },
                'basic_text': {
                    'model': ai_model,
                    'mode': 'default',
                    'system_message': 'You are an AI assistant specialized in extracting metadata from documents based on provided field definitions. For each field, analyze the document content and extract the corresponding value. CRITICALLY IMPORTANT: Respond for EACH field with a JSON object containing two keys: 1. "value": The extracted metadata value as a string. 2. "confidence": Your confidence level for this specific extraction, chosen from ONLY these three options: "High", "Medium", or "Low". Base your confidence on how certain you are about the extracted value given the document content and field definition. Example Response for a field: {"value": "INV-12345", "confidence": "High"}'
                }
            }
            items = [{'id': file_id, 'type': 'file'}]
            api_url = 'https://api.box.com/2.0/ai/extract_structured'
            request_body: Dict[str, Any] = {'items': items, 'ai_agent': ai_agent}

            if metadata_template:
                request_body['metadata_template'] = metadata_template
            elif fields:
                api_fields = []
                for field in fields:
                    if 'key' in field: # Already in correct API format
                        api_fields.append(field)
                    else: # Convert from internal format if necessary
                        api_field = {
                            'key': field.get('name', ''),
                            'displayName': field.get('display_name', field.get('name', '')),
                            'type': field.get('type', 'string') # Default to string if not provided
                        }
                        if 'description' in field and field['description']: # Ensure description is not empty
                            api_field['description'] = field['description']
                        if 'prompt' in field and field['prompt']: # Ensure prompt is not empty
                            api_field['prompt'] = field['prompt']
                        if field.get('type') == 'enum' and 'options' in field and field['options']: # Ensure options are not empty
                            api_field['options'] = field['options']
                        api_fields.append(api_field)
                request_body['fields'] = api_fields
            else:
                logger.error("Neither 'fields' nor 'metadata_template' provided for structured extraction.")
                raise ValueError('Either fields or metadata_template must be provided for structured extraction')

            logger.info(f'Making Box AI API call for structured extraction (file_id: {file_id}) with request body: {json.dumps(request_body)}')
            response = requests.post(api_url, headers=headers, json=request_body)

            if response.status_code != 200:
                logger.error(f'Box AI API error response: {response.text}')
                return {'error': f'Error in Box AI API call: {response.status_code} {response.reason}'}

            response_data = response.json()
            logger.info(f'Raw Box AI structured extraction response data: {json.dumps(response_data)}')

            processed_response: Dict[str, Any] = {}
            if 'answer' in response_data and isinstance(response_data['answer'], dict):
                answer_dict = response_data['answer']
                if 'fields' in answer_dict and isinstance(answer_dict['fields'], list):
                    logger.info("Processing 'answer' with 'fields' array format.")
                    fields_array = answer_dict['fields']
                    for field_item in fields_array:
                        if isinstance(field_item, dict) and 'key' in field_item and ('value' in field_item):
                            field_key = field_item['key']
                            extracted_value = field_item['value']
                            confidence_level = field_item.get('confidence', 'Low') # Default to Low
                            if not confidence_level or confidence_level not in ['High', 'Medium', 'Low']: # Check for None, empty, or invalid
                                logger.warning(f"Field {field_key}: AI provided confidence '{confidence_level}' which is invalid or missing. Defaulting to Low.")
                                confidence_level = 'Low'
                            processed_response[field_key] = extracted_value
                            processed_response[f'{field_key}_confidence'] = confidence_level
                        else:
                            logger.warning(f"Skipping invalid item in 'fields' array: {field_item}")
                else:
                    logger.info("Processing 'answer' as standard key-value dictionary.")
                    for field_key, field_data in answer_dict.items():
                        extracted_value = None
                        confidence_level = 'Low' # Default to Low initially

                        if isinstance(field_data, dict) and 'value' in field_data and 'confidence' in field_data:
                            extracted_value = field_data['value']
                            confidence_level = field_data['confidence']
                            if not confidence_level or confidence_level not in ['High', 'Medium', 'Low']:
                                logger.warning(f"Field {field_key} (file_id: {file_id}): AI provided confidence '{confidence_level}' which is invalid. Defaulting to Low.")
                                confidence_level = 'Low'
                        elif isinstance(field_data, dict) and 'value' in field_data: # Missing or invalid confidence
                            extracted_value = field_data['value']
                            # Confidence is already 'Low' by default
                            logger.warning(f"Field {field_key} (file_id: {file_id}): AI response provided 'value' but 'confidence' is missing or invalid: {field_data.get('confidence')}. Defaulting confidence to Low.")
                        elif field_data is None:
                            logger.info(f'Field {field_key} (file_id: {file_id}): Received null value. Setting value to None and confidence to Low.')
                            extracted_value = None
                            # Confidence is already 'Low' by default
                        else: # Any other format
                            logger.warning(f"Field {field_key} (file_id: {file_id}): Unexpected data format: {field_data}. Storing raw data and defaulting confidence to Low.")
                            extracted_value = field_data
                            # Confidence is already 'Low' by default

                        processed_response[field_key] = extracted_value
                        processed_response[f'{field_key}_confidence'] = confidence_level

            elif 'answer' in response_data and isinstance(response_data['answer'], str):
                logger.info(f"Processing 'answer' as string (potential freeform JSON) for file_id: {file_id}.")
                response_text = response_data['answer']
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        parsed_json = json.loads(json_str)
                        if isinstance(parsed_json, dict):
                            for field_key, field_data in parsed_json.items():
                                if isinstance(field_data, dict) and 'value' in field_data and ('confidence' in field_data):
                                    extracted_value = field_data['value']
                                    confidence_level = field_data['confidence']
                                    if not confidence_level or confidence_level not in ['High', 'Medium', 'Low']:
                                        logger.warning(f"Field {field_key} (file_id: {file_id}): AI provided confidence '{confidence_level}' from parsed string which is invalid. Defaulting to Low.")
                                        confidence_level = 'Low'
                                    processed_response[field_key] = extracted_value
                                    processed_response[f'{field_key}_confidence'] = confidence_level
                                else:
                                    logger.warning(f"Field {field_key} (file_id: {file_id}): Parsed JSON from AI 'answer' string for this field did not contain 'value'/'confidence' dict: {field_data}. Defaulting confidence to Low.")
                                    processed_response[field_key] = field_data
                                    processed_response[f'{field_key}_confidence'] = 'Low'
                        else:
                            logger.warning(f"Parsed JSON from 'answer' string is not a dictionary (file_id: {file_id}): {parsed_json}")
                            processed_response['_raw_response'] = response_text
                            processed_response['_confidence_processing_failed'] = True # Mark that confidence processing might be incomplete
                    else:
                        logger.warning(f"No JSON object found in 'answer' string (file_id: {file_id}).")
                        processed_response['_raw_response'] = response_text
                        processed_response['_confidence_processing_failed'] = True # Mark that confidence processing might be incomplete
                except Exception as e:
                    logger.error(f'Error parsing JSON from answer string (file_id: {file_id}): {str(e)}')
                    processed_response['_raw_response'] = response_text
                    processed_response['_confidence_processing_failed'] = True # Mark that confidence processing might be incomplete
            elif 'entries' in response_data and len(response_data['entries']) > 0:
                logger.info(f"Processing response using fallback 'entries' format for file_id: {file_id}.")
                entry = response_data['entries'][0]
                if 'metadata' in entry:
                    metadata = entry['metadata']
                    for field_key, field_value in metadata.items():
                        extracted_value = field_value
                        confidence_level = 'Low' # Default confidence
                        try:
                            if isinstance(field_value, str) and field_value.strip().startswith('{') and field_value.strip().endswith('}'):
                                try:
                                    parsed_value = json.loads(field_value)
                                    if isinstance(parsed_value, dict) and 'value' in parsed_value and ('confidence' in parsed_value):
                                        extracted_value = parsed_value['value']
                                        confidence_level = parsed_value['confidence']
                                        if not confidence_level or confidence_level not in ['High', 'Medium', 'Low']:
                                            logger.warning(f"Field {field_key} (file_id: {file_id}): AI provided confidence '{confidence_level}' in 'entries' path which is invalid. Defaulting to Low.")
                                            confidence_level = 'Low'
                                    else:
                                        logger.warning(f"Field {field_key} (file_id: {file_id}): Parsed JSON but keys 'value' and 'confidence' not found. Using raw value.")
                                        # confidence_level remains 'Low' (initial default)
                                except json.JSONDecodeError:
                                    logger.warning(f"Field {field_key} (file_id: {file_id}): Failed to parse potential JSON value '{field_value}'. Using raw value.")
                                    # confidence_level remains 'Low' (initial default)
                            else:
                                # Value is not a JSON string, use as is with Low confidence
                                logger.info(f'Field {field_key} (file_id: {file_id}): Value is not the expected JSON format. Using raw value and Low confidence.')
                            processed_response[field_key] = extracted_value
                            processed_response[f'{field_key}_confidence'] = confidence_level
                        except Exception as e: # Catch any error during field processing
                            logger.error(f"Error processing field {field_key} (file_id: {file_id}) with value '{field_value}': {str(e)}")
                            processed_response[field_key] = field_value # Store raw data on error
                            processed_response[f'{field_key}_confidence'] = 'Low' # Default confidence on error
                else:
                    logger.warning(f"No 'metadata' field found in the structured API entry (file_id: {file_id}): {entry}")
                    processed_response['_error'] = "No 'metadata' field in API entry"
                    if '_confidence_processing_failed' not in processed_response: processed_response['_confidence_processing_failed'] = True
            else:
                logger.warning(f"Neither 'answer' nor 'entries' field found in the structured API response (file_id: {file_id}): {response_data}")
                processed_response['_error'] = "Neither 'answer' nor 'entries' field in API response"
                if '_confidence_processing_failed' not in processed_response: processed_response['_confidence_processing_failed'] = True
            return processed_response
        except Exception as e:
            logger.error(f'Error in structured metadata extraction call for file_id {file_id}: {str(e)}', exc_info=True)
            return {'error': str(e), '_confidence_processing_failed': True}

    def extract_freeform_metadata(client: Any, file_id: str, prompt: str, ai_model: str = 'azure__openai__gpt_4o_mini') -> Dict[str, Any]:
        """
        Extract freeform metadata from a file using Box AI API
        
        Args:
            client (Any): The Box API client.
            file_id (str): Box file ID
            prompt (str): Extraction prompt
            ai_model (str): AI model to use for extraction
            
        Returns:
            dict: Extracted metadata with confidence scores
        """
        logger.info(f"Starting freeform metadata extraction for file_id: {file_id} using AI model: {ai_model}")
        try:
            # client = st.session_state.client # Client is now passed as an argument
            access_token = None
            if hasattr(client, '_oauth'):
                access_token = client._oauth.access_token
            elif hasattr(client, 'auth') and hasattr(client.auth, 'access_token'):
                access_token = client.auth.access_token
            if not access_token:
                logger.error(f"Could not retrieve access token from client (file_id: {file_id}).")
                raise ValueError('Could not retrieve access token from client')

            headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
            
            enhanced_prompt = prompt
            # Ensure prompt asks for confidence if not already present
            # Simple check, could be made more robust
            if not ('confidence' in prompt.lower() and 'json' in prompt.lower()):
                logger.info(f"Enhancing prompt for file_id: {file_id} to explicitly request JSON with value/confidence.")
                enhanced_prompt = prompt + " For each extracted field, provide your confidence level (High, Medium, or Low) in the accuracy of the extraction. Format your response as a JSON object where each key is the field name, and the value is another JSON object with two keys: 'value' (the extracted information) and 'confidence' (your confidence level). Example: { \"InvoiceNumber\": { \"value\": \"INV-123\", \"confidence\": \"High\" }, \"TotalAmount\": { \"value\": \"$500.00\", \"confidence\": \"Medium\" } }"
            else:
                logger.info(f"Prompt for file_id: {file_id} already seems to request confidence and JSON.")


            ai_agent = {
                'type': 'ai_agent_text_gen', # Verified as per requirement
                'basic_text': {
                    'model': ai_model,
                    'prompt': enhanced_prompt,
                    'system_message': 'You are an AI assistant that extracts information from documents and returns it as a JSON object. For each field, provide a value and a confidence level (High, Medium, or Low).'
                }
            }
            items = [{'id': file_id, 'type': 'file'}] # Corrected item structure
            api_url = 'https://api.box.com/2.0/ai/text_gen'
            request_body = {'items': items, 'ai_agent': ai_agent}

            logger.info(f'Making Box AI API call for freeform extraction (file_id: {file_id}) with request body: {json.dumps(request_body)}')
            response = requests.post(api_url, headers=headers, json=request_body)

            if response.status_code != 200:
                logger.error(f'Box AI API error response (file_id: {file_id}): {response.status_code} {response.reason} - {response.text}')
                return {'error': f'Error in Box AI API call: {response.status_code} {response.reason}', '_confidence_processing_failed': True}

            response_data = response.json() # Assuming this is always a dict if status is 200
            logger.info(f'Raw Box AI freeform extraction response data (file_id: {file_id}): {json.dumps(response_data)}')

            processed_response: Dict[str, Any] = {}
            # Prioritize 'answer' directly from response_data
            answer_text = response_data.get('answer')

            if not answer_text and 'entries' in response_data and len(response_data['entries']) > 0 and 'answer' in response_data['entries'][0]:
                logger.info(f"Using 'answer' from 'entries' fallback for file_id: {file_id}.")
                answer_text = response_data['entries'][0]['answer']

            if answer_text and isinstance(answer_text, str):
                try:
                    json_start = answer_text.find('{')
                    json_end = answer_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = answer_text[json_start:json_end]
                        parsed_json = json.loads(json_str)
                        if isinstance(parsed_json, dict):
                            for key, value_confidence_pair in parsed_json.items():
                                extracted_val = None
                                confidence_val = 'Low' # Default confidence

                                if isinstance(value_confidence_pair, dict) and 'value' in value_confidence_pair and 'confidence' in value_confidence_pair:
                                    extracted_val = value_confidence_pair['value']
                                    confidence_val = value_confidence_pair['confidence']
                                    if not confidence_val or confidence_val not in ['High', 'Medium', 'Low']:
                                        logger.warning(f"Field {key} (file_id: {file_id}): AI provided confidence '{confidence_val}' in freeform response which is invalid. Defaulting to Low.")
                                        confidence_val = 'Low'
                                elif isinstance(value_confidence_pair, dict) and 'value' in value_confidence_pair: # Value present, confidence missing/invalid
                                    extracted_val = value_confidence_pair['value']
                                    logger.warning(f"Field {key} (file_id: {file_id}): Freeform response for field has 'value' but missing or invalid 'confidence': {value_confidence_pair.get('confidence')}. Defaulting confidence to Low.")
                                    # confidence_val remains 'Low'
                                else: # Does not match expected dict structure {value: X, confidence: Y}
                                    logger.warning(f"Field {key} (file_id: {file_id}): Unexpected format for value_confidence_pair in freeform response: {value_confidence_pair}. Storing as is, defaulting confidence to Low.")
                                    extracted_val = value_confidence_pair # Store the raw pair/value

                                processed_response[key] = extracted_val
                                processed_response[f'{key}_confidence'] = confidence_val
                        else:
                            logger.warning(f"Parsed JSON from 'answer' string is not a dictionary (file_id: {file_id}): {parsed_json}. Storing raw answer.")
                            processed_response['_raw_answer'] = answer_text
                            if '_confidence_processing_failed' not in processed_response: processed_response['_confidence_processing_failed'] = True
                    else:
                        logger.warning(f"No JSON object found in 'answer' string (file_id: {file_id}). Storing raw answer: {answer_text}")
                        processed_response['_raw_answer'] = answer_text
                        if '_confidence_processing_failed' not in processed_response: processed_response['_confidence_processing_failed'] = True
                except json.JSONDecodeError as e_json:
                    logger.error(f'Error parsing JSON from freeform answer string (file_id: {file_id}): {str(e_json)}. Raw answer: {answer_text}')
                    processed_response['_raw_answer'] = answer_text
                    processed_response['_error_parsing_json'] = str(e_json)
                    if '_confidence_processing_failed' not in processed_response: processed_response['_confidence_processing_failed'] = True
            else:
                logger.warning(f"No usable 'answer' string found in the freeform API response (file_id: {file_id}): {response_data}")
                processed_response['_error'] = "No 'answer' string in API response"
                if '_confidence_processing_failed' not in processed_response: processed_response['_confidence_processing_failed'] = True

            return processed_response
        except Exception as e:
            logger.error(f'Error in freeform metadata extraction call for file_id {file_id}: {str(e)}', exc_info=True)
            return {'error': str(e), '_confidence_processing_failed': True}

    # Return the dictionary of functions
    return {
        'structured': extract_structured_metadata,
        'freeform': extract_freeform_metadata
    }

# Example of how it might be called (for testing, not part of the module's direct execution)
if __name__ == '__main__':
    # This part is for testing and won't run when imported
    class MockOAuth:
        def __init__(self, token):
            self.access_token = token

    class MockClient:
        def __init__(self, token):
            self._oauth = MockOAuth(token)
            # self.auth = MockOAuth(token) # Alternative way to store auth

    # Simulate Streamlit session state for testing
    st.session_state.client = MockClient("test_access_token")

    functions = get_extraction_functions()
    print(f"Available extraction functions: {list(functions.keys())}")

    # Mock a call (won't actually make an API request without a real token and file)
    # test_file_id = "12345"
    # test_prompt = "Extract the invoice number and total amount."
    # if 'freeform' in functions:
    #     result = functions['freeform'](client=st.session_state.client, file_id=test_file_id, prompt=test_prompt)
    #     print(f"Mock freeform call result: {result}")

    # test_fields = [{'key': 'invoice_number', 'displayName': 'Invoice Number', 'type': 'string'}]
    # if 'structured' in functions:
    #     result_structured = functions['structured'](client=st.session_state.client, file_id=test_file_id, fields=test_fields)
    #     print(f"Mock structured call result: {result_structured}")

