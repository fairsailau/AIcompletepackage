"""
Box AI Tool for LangChain integration.

This module provides a custom LangChain tool that interfaces with Box AI API
for document processing, categorization, and metadata extraction.
"""

import logging
import json
import time
import re # Added import
import requests # Moved import to top
from typing import Dict, Any, List, Optional, Union, Callable, Type

# LangChain imports
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BoxAIQueryInput(BaseModel):
    """Input for Box AI query tool."""
    query: str = Field(..., description="The query to send to Box AI")
    file_id: str = Field(..., description="The Box file ID to process")
    query_type: str = Field("categorization", description="Type of query: categorization, metadata, or general")

class BoxAITool(BaseTool):
    """Tool for querying Box AI API."""
    
    name: str = "box_ai_query"
    description: str = "Query Box AI to process documents, categorize them, or extract metadata"
    args_schema: Type[BaseModel] = BoxAIQueryInput
    box_client: Any = None # Field declaration
    
    def __init__(self, box_client: Any): # Type hint for parameter
        """Initialize the Box AI Tool with a Box client."""
        super().__init__()
        self.box_client = box_client
    
    def _run(
        self, 
        query: str, 
        file_id: str, 
        query_type: str = "categorization",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Run Box AI query on documents."""
        try:
            logger.info(f"Running Box AI query on file {file_id}: {query}")
            
            # Get access token from Box client
            access_token = None
            if hasattr(self.box_client, '_oauth'):
                access_token = self.box_client._oauth.access_token
            elif hasattr(self.box_client, 'auth') and hasattr(self.box_client.auth, 'access_token'):
                access_token = self.box_client.auth.access_token
            
            if not access_token:
                return json.dumps({
                    "success": False,
                    "error": "Could not retrieve Box access token"
                })
            
            # Prepare headers for Box API
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Process based on query type
            if query_type == "categorization":
                result = self._run_categorization(file_id, query, headers)
            elif query_type == "metadata":
                result = self._run_metadata_extraction(file_id, query, headers)
            else:
                result = self._run_general_query(file_id, query, headers)
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error in Box AI query: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def _run_categorization(self, file_id: str, query: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Run document categorization using Box AI /ai/ask endpoint."""
        try:
            api_url = 'https://api.box.com/2.0/ai/ask' # Changed API URL to /ai/ask
            
            # Parse categories from the original query parameter or use defaults
            # The 'query' parameter to _run_categorization is the original query from the tool input.
            categories_from_query = []
            if "categories:" in query.lower():
                categories_text = query.lower().split("categories:")[1].strip()
                categories_from_query = [cat.strip() for cat in categories_text.split(",")]
            
            if not categories_from_query:
                categories_from_query = [
                    "Sales Contract", "Invoice", "Tax Document", "Financial Report",
                    "Employment Contract", "PII Document", "Other"
                ] # Simplified default list for the prompt

            # Construct the prompt for the Q&A endpoint
            categories_list_str = "\n".join([f"- {cat_name}" for cat_name in categories_from_query])
            prompt = f"""Please analyze this document and categorize it into one of the following categories:
{categories_list_str}

Respond ONLY in the following format (exactly three lines):
Category: [selected category name]
Confidence: [confidence score between 0.0 and 1.0, e.g., 0.75]
Reasoning: [Your detailed reasoning for the categorization]
"""
            # Prepare request body for /ai/ask
            request_body = {
                'mode': 'question_answering',
                'prompt': prompt, # 'prompt' is the variable already holding the constructed categorization question
                'items': [{'type': 'file', 'id': file_id}]
            }
            
            response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
            
            if response.status_code == 200:
                response_data = response.json()
                answer_text = response_data.get('answer') # Get answer directly from /ai/ask response
                
                if answer_text is not None:
                    # Parse the answer_text (existing logic should be mostly fine)
                    parsed_category = "Other"
                    parsed_confidence = 0.0
                    parsed_reasoning = "No reasoning provided by AI or parsing failed."

                    cat_match = re.search(r"Category: (.*)", answer_text)
                    if cat_match:
                        parsed_category = cat_match.group(1).strip()

                    conf_match = re.search(r"Confidence: ([\d.]+)", answer_text)
                    if conf_match:
                        try:
                            parsed_confidence = float(conf_match.group(1).strip())
                        except ValueError:
                            logger.warning(f"Could not parse confidence from AI response: {conf_match.group(1)} for file {file_id}")
                            parsed_confidence = 0.1 # Default to low if parsing fails

                    # Try to get reasoning, ensuring it's not part of Category or Confidence lines
                    reasoning_lines = []
                    for line in answer_text.split('\n'):
                        if line.startswith("Reasoning:"):
                            reasoning_lines.append(line.replace("Reasoning:", "").strip())
                        elif not line.startswith("Category:") and not line.startswith("Confidence:"):
                            reasoning_lines.append(line.strip())

                    if reasoning_lines:
                        parsed_reasoning = "\n".join(reasoning_lines).strip()
                    elif cat_match and conf_match : # If only cat and conf found, but no explicit Reasoning: line
                            # Check if there's text after the confidence line
                        answer_after_conf = answer_text.split(conf_match.group(0),1)[-1].strip()
                        if answer_after_conf:
                            parsed_reasoning = answer_after_conf

                    # Ensure the parsed category is one of the requested categories, otherwise set to "Other"
                    if parsed_category not in categories_from_query and "Other" in categories_from_query:
                        logger.warning(f"AI returned category '{parsed_category}' not in requested list for file {file_id}. Defaulting to 'Other'.")
                        parsed_category = "Other"
                    elif parsed_category not in categories_from_query:
                            logger.warning(f"AI returned category '{parsed_category}' not in requested list for file {file_id} and 'Other' not available. Keeping AI category.")


                    return {
                        "success": True,
                        "document_type": parsed_category,
                        "confidence": parsed_confidence,
                        "reasoning": parsed_reasoning
                    }
                else:
                    logger.warning(f"Categorization (via /ai/ask) for file {file_id} missing 'answer' in response: {response_data}")
                    return {"success": False, "error": "Malformed response from Box AI /ai/ask (missing answer field)", "response_data": response_data}
            # Handle non-200 error or unexpected response
            logger.error(f"Box AI categorization (via /ai/ask) failed for file {file_id}: {response.status_code}, Response: {response.text[:500]}")
            return {
                "success": False,
                "error": f"Box AI categorization (via /ai/ask) failed: Status {response.status_code}",
                "response_text": response.text[:500] if response.text else "No response text"
            }
            
        except requests.exceptions.RequestException as req_e:
            logger.error(f"RequestException during Box AI categorization (via /ai/ask) for file {file_id}: {str(req_e)}")
            return {"success": False, "error": f"RequestException: {str(req_e)}"}
        except Exception as e:
            logger.error(f"Error in Box AI categorization (via /ai/ask) for file {file_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_metadata_extraction(self, file_id: str, query: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Run metadata extraction using Box AI."""
        try:
            # Use Box AI API to extract metadata
            api_url = 'https://api.box.com/2.0/ai/text_extraction'
            
            # Parse fields from query if provided
            fields = []
            if "fields:" in query.lower():
                fields_text = query.lower().split("fields:")[1].strip()
                fields = [field.strip() for field in fields_text.split(",")]
            
            # Default fields if not specified
            if not fields:
                fields = [
                    "invoice_number",
                    "invoice_date",
                    "due_date",
                    "vendor_name",
                    "total_amount"
                ]
            
            # Prepare request body
            request_body = {
                'items': [{'id': file_id, 'type': 'file'}],
                'task': {
                    'type': 'key_value_extraction',
                    'keys': fields
                }
            }
            
            # Make API request
            response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract metadata result
                if 'entries' in response_data and len(response_data['entries']) > 0:
                    entry = response_data['entries'][0]
                    if 'key_value_extraction' in entry:
                        extractions = entry['key_value_extraction']
                        
                        # Format result
                        result = {
                            "success": True,
                            "metadata": {}
                        }
                        
                        # Process each extracted field
                        for extraction in extractions:
                            key = extraction.get('key', '')
                            value = extraction.get('value', '')
                            confidence = extraction.get('confidence', 0.0)
                            
                            # Add to result
                            result["metadata"][key] = value
                            result["metadata"][f"{key}_confidence"] = self._confidence_to_category(confidence)
                            result["metadata"][f"{key}_confidence_numeric"] = confidence
                        
                        return result
            
            # Handle error or unexpected response
            return {
                "success": False,
                "error": f"Box AI metadata extraction failed: {response.status_code}",
                "response": response.text if response.text else "No response"
            }
            
        except Exception as e:
            logger.error(f"Error in Box AI metadata extraction: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_general_query(self, file_id: str, query: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Run general query using Box AI."""
        try:
            # Use Box AI API for general query
            api_url = 'https://api.box.com/2.0/ai/text_qa'
            
            # Prepare request body
            request_body = {
                'items': [{'id': file_id, 'type': 'file'}],
                'task': {
                    'type': 'question_answering',
                    'question': query
                }
            }
            
            # Make API request
            response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract answer
                if 'entries' in response_data and len(response_data['entries']) > 0:
                    entry = response_data['entries'][0]
                    if 'question_answering' in entry:
                        answer = entry['question_answering'].get('answer', '')
                        confidence = entry['question_answering'].get('confidence', 0.0)
                        
                        # Format result
                        result = {
                            "success": True,
                            "answer": answer,
                            "confidence": confidence
                        }
                        
                        return result
            
            # Handle error or unexpected response
            return {
                "success": False,
                "error": f"Box AI general query failed: {response.status_code}",
                "response": response.text if response.text else "No response"
            }
            
        except Exception as e:
            logger.error(f"Error in Box AI general query: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _confidence_to_category(self, confidence: float) -> str:
        """Convert numeric confidence to category (High, Medium, Low)."""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        else:
            return "Low"

class BoxAIDocumentProcessor:
    """
    Processor for handling document processing with Box AI.
    This class provides higher-level functions for document processing workflows.
    """
    
    def __init__(self, box_client):
        """Initialize the processor with a Box client."""
        self.box_client = box_client
        self.box_ai_tool = BoxAITool(box_client)
    
    def categorize_document(self, file_id: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Categorize a document using Box AI.
        
        Args:
            file_id: Box file ID
            categories: Optional list of categories to consider
            
        Returns:
            Categorization result with confidence
        """
        # Prepare query for BoxAITool's _run_categorization method
        if categories:
            # Format so that _run_categorization can parse it via "categories:"
            query = f"categories: {', '.join(categories)}"
        else:
            # If no categories are provided, _run_categorization will use its defaults.
            # Sending a simple query string is fine as _run_categorization's
            # category parsing is conditional on "categories:" being present.
            query = "Categorize this document."
        
        # Run query
        result_json = self.box_ai_tool._run(
            query=query,
            file_id=file_id,
            query_type="categorization"
        )
        
        # Parse result
        try:
            result = json.loads(result_json)
            return result
        except Exception as e:
            logger.error(f"Error parsing categorization result: {str(e)}")
            return {
                "success": False,
                "error": f"Error parsing result: {str(e)}"
            }
    
    def extract_metadata(self, file_id: str, fields: List[str]) -> Dict[str, Any]:
        """
        Extract metadata from a document using Box AI.
        
        Args:
            file_id: Box file ID
            fields: List of fields to extract
            
        Returns:
            Extracted metadata with confidence
        """
        # Prepare query
        query = f"Extract these fields: {', '.join(fields)}"
        
        # Run query
        result_json = self.box_ai_tool._run(
            query=query,
            file_id=file_id,
            query_type="metadata"
        )
        
        # Parse result
        try:
            result = json.loads(result_json)
            return result
        except Exception as e:
            logger.error(f"Error parsing metadata result: {str(e)}")
            return {
                "success": False,
                "error": f"Error parsing result: {str(e)}"
            }
    
    def ask_question(self, file_id: str, question: str) -> Dict[str, Any]:
        """
        Ask a question about a document using Box AI.
        
        Args:
            file_id: Box file ID
            question: Question to ask
            
        Returns:
            Answer with confidence
        """
        # Run query
        result_json = self.box_ai_tool._run(
            query=question,
            file_id=file_id,
            query_type="general"
        )
        
        # Parse result
        try:
            result = json.loads(result_json)
            return result
        except Exception as e:
            logger.error(f"Error parsing question result: {str(e)}")
            return {
                "success": False,
                "error": f"Error parsing result: {str(e)}"
            }
    
    def process_document_workflow(
        self,
        file_id: str,
        categories: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a document through a complete workflow of categorization and metadata extraction.
        
        Args:
            file_id: Box file ID
            categories: Optional list of categories for categorization
            metadata_fields: Optional list of fields for metadata extraction
            
        Returns:
            Complete processing result
        """
        result = {
            "file_id": file_id,
            "success": True,
            "categorization": None,
            "metadata": None,
            "overall_confidence": 0.0,
            "processing_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # Get file info
            file_info = self.box_client.file(file_id).get()
            result["file_name"] = file_info.name
            
            # Step 1: Categorize document
            cat_result = self.categorize_document(file_id, categories)
            if cat_result.get("success", False):
                result["categorization"] = {
                    "document_type": cat_result.get("document_type", "Other"),
                    "confidence": cat_result.get("confidence", 0.0),
                    "reasoning": cat_result.get("reasoning", "")
                }
            else:
                result["categorization"] = {
                    "document_type": "Error",
                    "confidence": 0.0,
                    "reasoning": cat_result.get("error", "Unknown error")
                }
                result["success"] = False
            
            # Step 2: Extract metadata if fields provided
            if metadata_fields and result["success"]:
                meta_result = self.extract_metadata(file_id, metadata_fields)
                if meta_result.get("success", False):
                    result["metadata"] = meta_result.get("metadata", {})
                else:
                    result["metadata"] = {
                        "error": meta_result.get("error", "Unknown error")
                    }
                    result["success"] = False
            
            # Calculate overall confidence
            if result["categorization"] and "confidence" in result["categorization"]:
                cat_confidence = result["categorization"]["confidence"]
                
                # If metadata was extracted, include its confidence
                if result["metadata"]:
                    # Calculate average metadata confidence
                    meta_confidences = [
                        value for key, value in result["metadata"].items()
                        if key.endswith("_confidence_numeric") and isinstance(value, (int, float))
                    ]
                    
                    if meta_confidences:
                        meta_confidence = sum(meta_confidences) / len(meta_confidences)
                        # Overall confidence is weighted average
                        result["overall_confidence"] = (cat_confidence * 0.6) + (meta_confidence * 0.4)
                    else:
                        result["overall_confidence"] = cat_confidence
                else:
                    result["overall_confidence"] = cat_confidence
            
        except Exception as e:
            logger.error(f"Error in document workflow: {str(e)}")
            result["success"] = False
            result["error"] = str(e)
        
        # Record processing time
        result["processing_time"] = time.time() - start_time
        
        return result

def create_box_ai_langchain_tools(box_client) -> List[BaseTool]:
    """
    Create LangChain tools for Box AI integration.
    
    Args:
        box_client: Box client instance
        
    Returns:
        List of LangChain tools
    """
    # Create Box AI processor
    processor = BoxAIDocumentProcessor(box_client)
    
    # Create tools
    tools = [
        # Box AI query tool (low-level)
        BoxAITool(box_client),
        
        # Document categorization tool
        Tool(
            name="categorize_document_with_box_ai",
            description="Categorize a document using Box AI",
            func=lambda file_id, categories=None: json.dumps(
                processor.categorize_document(file_id, categories)
            )
        ),
        
        # Metadata extraction tool
        Tool(
            name="extract_metadata_with_box_ai",
            description="Extract metadata from a document using Box AI",
            func=lambda file_id, fields: json.dumps(
                processor.extract_metadata(file_id, fields)
            )
        ),
        
        # Document question tool
        Tool(
            name="ask_document_question_with_box_ai",
            description="Ask a question about a document using Box AI",
            func=lambda file_id, question: json.dumps(
                processor.ask_question(file_id, question)
            )
        ),
        
        # Complete workflow tool
        Tool(
            name="process_document_workflow_with_box_ai",
            description="Process a document through a complete workflow using Box AI",
            func=lambda file_id, categories=None, metadata_fields=None: json.dumps(
                processor.process_document_workflow(file_id, categories, metadata_fields)
            )
        )
    ]
    
    return tools
