"""
LangChain Agent with Box AI integration for autonomous document processing.

This module implements a LangChain agent that uses Box AI for document processing,
with intelligent human-in-the-loop escalation based on confidence boundaries.
"""

import streamlit as st
import pandas as pd
import logging
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI

# Import Box AI Tool
from modules.box_ai_tool import (
    BoxAITool,
    BoxAIDocumentProcessor,
    create_box_ai_langchain_tools
)
from modules.processing import get_fields_for_ai_from_template # Added import

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Status of document processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    AUTO_APPROVED = "auto_approved"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class ProcessingBoundaries:
    """Configuration for processing boundaries and thresholds."""
    # Confidence thresholds
    auto_approve_threshold: float = 0.85
    human_review_threshold: float = 0.6
    auto_reject_threshold: float = 0.3
    
    # Processing limits
    max_files_per_batch: int = 50
    max_processing_time_minutes: int = 30
    
    # Escalation criteria
    escalate_on_category_change: bool = True
    escalate_on_low_metadata_confidence: bool = True
    escalate_on_field_inconsistency: bool = True
    
    # Human review timeout
    human_review_timeout_hours: int = 24

@dataclass
class DocumentProcessingResult:
    """Result of document processing by the agent."""
    file_id: str
    file_name: str
    status: ProcessingStatus
    categorization_result: Optional[Dict[str, Any]] = None
    metadata_result: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    escalation_reason: Optional[str] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    human_feedback: Optional[Dict[str, Any]] = None

class BoxAIDocumentProcessingAgent:
    """LangChain agent for autonomous document processing using Box AI."""
    
    def __init__(
        self,
        client,
        boundaries: ProcessingBoundaries,
        llm_model: str = "gpt-4",
        enable_human_loop: bool = True,
        openai_api_key: Optional[str] = None # New parameter
    ):
        self.client = client
        self.boundaries = boundaries
        self.enable_human_loop = enable_human_loop
        
        # Initialize Box AI processor
        self.box_ai_processor = BoxAIDocumentProcessor(client)
        
        # Initialize LLM for agent reasoning (not for document processing)
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.1,
            max_tokens=2000,
            api_key=openai_api_key # Pass the API key
        )

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Processing state
        self.processing_queue: List[str] = []
        self.processing_results: Dict[str, DocumentProcessingResult] = {}
        self.human_review_queue: List[str] = []
        
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the LangChain agent."""
        
        # Get Box AI tools
        box_ai_tools = create_box_ai_langchain_tools(self.client)
        
        # Create additional tools for agent workflow
        workflow_tools = [
            Tool(
                name="check_confidence_boundaries",
                description="Check if confidence scores meet processing boundaries",
                func=self._check_confidence_boundaries
            ),
            Tool(
                name="escalate_to_human",
                description="Escalate a document to human review with reason",
                func=self._escalate_to_human
            ),
            Tool(
                name="get_file_info",
                description="Get basic information about a file",
                func=self._get_file_info
            )
        ]
        
        # Combine all tools
        return box_ai_tools + workflow_tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent."""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent document processing agent that uses Box AI to autonomously process documents through categorization and metadata extraction workflows.

Your responsibilities:
1. Process documents through categorization and metadata extraction using Box AI
2. Evaluate confidence scores against defined boundaries
3. Make autonomous decisions for high-confidence results
4. Escalate to human review when confidence is below thresholds
5. Provide clear reasoning for all decisions

Processing Boundaries:
- Auto-approve threshold: {auto_approve_threshold}
- Human review threshold: {human_review_threshold}
- Auto-reject threshold: {auto_reject_threshold}

Escalation Criteria:
- Low confidence scores
- Category changes during processing
- Field inconsistencies in metadata
- Any processing errors

Always provide detailed reasoning for your decisions and be conservative when in doubt.
""".format(
                auto_approve_threshold=self.boundaries.auto_approve_threshold,
                human_review_threshold=self.boundaries.human_review_threshold,
                auto_reject_threshold=self.boundaries.auto_reject_threshold
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=10,
            early_stopping_method="generate"
        )
        
        return agent_executor
    
    def _check_confidence_boundaries(self, confidence_scores_json: str) -> str:
        """Check if confidence scores meet processing boundaries."""
        try:
            scores = json.loads(confidence_scores_json)
            
            # Check categorization confidence
            cat_confidence = scores.get("categorization_confidence", 0.0)
            
            # Check metadata confidence (average of all fields)
            metadata_confidences = scores.get("metadata_confidences", {})
            avg_metadata_confidence = sum(metadata_confidences.values()) / len(metadata_confidences) if metadata_confidences else 0.0
            
            # Determine action based on boundaries
            if cat_confidence >= self.boundaries.auto_approve_threshold and avg_metadata_confidence >= self.boundaries.auto_approve_threshold:
                action = "auto_approve"
            elif cat_confidence >= self.boundaries.human_review_threshold or avg_metadata_confidence >= self.boundaries.human_review_threshold:
                action = "human_review"
            else:
                action = "auto_reject"
            
            return json.dumps({
                "action": action,
                "categorization_confidence": cat_confidence,
                "metadata_confidence": avg_metadata_confidence,
                "reasoning": f"Categorization: {cat_confidence:.2f}, Metadata: {avg_metadata_confidence:.2f}"
            })
            
        except Exception as e:
            logger.error(f"Error checking confidence boundaries: {str(e)}")
            return json.dumps({
                "action": "error",
                "error": str(e)
            })
    
    def _escalate_to_human(self, file_id: str, reason: str, data_json: str) -> str:
        """Escalate a document to human review."""
        try:
            # Parse data
            data = json.loads(data_json) if data_json else {}
            
            # Add to human review queue
            self.human_review_queue.append(file_id)
            
            # Update processing result
            if file_id in self.processing_results:
                self.processing_results[file_id].status = ProcessingStatus.HUMAN_REVIEW_REQUIRED
                self.processing_results[file_id].escalation_reason = reason
            
            logger.info(f"Escalated file {file_id} to human review: {reason}")
            
            return json.dumps({
                "success": True,
                "message": f"File {file_id} escalated to human review",
                "reason": reason
            })
            
        except Exception as e:
            logger.error(f"Error escalating {file_id} to human: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def _get_file_info(self, file_id: str) -> str:
        """Get information about a file."""
        try:
            file_info = self.client.file(file_id).get()
            return json.dumps({
                "success": True,
                "file_name": file_info.name,
                "file_size": file_info.size,
                "file_type": file_info.name.split('.')[-1] if '.' in file_info.name else "unknown"
            })
        except Exception as e:
            logger.error(f"Error getting file info for {file_id}: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def process_document(
        self,
        file_id: str,
        field_definitions: List[Dict[str, Any]],
        categories: Optional[List[str]] = None
    ) -> DocumentProcessingResult:
        """Process a single document through the agent workflow."""
        
        start_time = time.time()
        
        # Initialize result
        result = DocumentProcessingResult(
            file_id=file_id,
            file_name=f"File {file_id}",
            status=ProcessingStatus.PROCESSING
        )
        
        try:
            # Get file info
            file_info_response = self._get_file_info(file_id)
            file_info = json.loads(file_info_response)
            if file_info.get("success", False):
                result.file_name = file_info.get("file_name", f"File {file_id}")
            
            # Extract field names for metadata extraction
            field_names = [field.get("name") for field in field_definitions if "name" in field]
            logger.info(f"File {file_id}: Pre-metadata extraction. result.success: {result.success}. Metadata fields to attempt: {field_names if field_names else 'None'}") # Log field_names

            # Extract field names for metadata extraction
            field_names = [field.get("name") for field in field_definitions if "name" in field]
            logger.info(f"File {file_id}: Pre-workflow. Metadata fields to attempt: {field_names if field_names else 'None'}")

            # Process document using Box AI
            processing_result = self.box_ai_processor.process_document_workflow(
                file_id=file_id,
                categories=categories,
                metadata_fields=field_names
            )
            logger.info(f"File {file_id}: Agent received processing_result from workflow: {processing_result}")

            # Local flag based on the tool's own success report
            tool_execution_successful = processing_result.get("success", False)

            # Initialize confidence_scores if it's None
            if result.confidence_scores is None:
                result.confidence_scores = {"overall": 0.0, "categorization": 0.0, "metadata": {}}
            else: # Ensure keys exist
                result.confidence_scores.setdefault("overall", 0.0)
                result.confidence_scores.setdefault("categorization", 0.0)
                result.confidence_scores.setdefault("metadata", {})

            # Handle categorization part
            categorization_details_from_workflow = processing_result.get("categorization", {})
            if tool_execution_successful and categorization_details_from_workflow.get("document_type") != "Error" and categorization_details_from_workflow.get("success", True if categorization_details_from_workflow else False) :
                retrieved_doc_type = categorization_details_from_workflow.get("document_type", "Other")
                retrieved_confidence = categorization_details_from_workflow.get("confidence", 0.0)
                retrieved_reasoning = categorization_details_from_workflow.get("reasoning", "")

                result.categorization_result = {
                    "document_type": retrieved_doc_type,
                    "confidence": retrieved_confidence,
                    "reasoning": retrieved_reasoning
                }
                result.confidence_scores["categorization"] = retrieved_confidence
                logger.info(f"File {file_id}: Workflow Categorization Part SUCCEEDED. Type: '{retrieved_doc_type}', Conf: {retrieved_confidence:.2f}")
            else:
                tool_execution_successful = False # Mark as failed if cat part specifically errored or tool indicated failure
                error_reason = categorization_details_from_workflow.get("reasoning", categorization_details_from_workflow.get("error", processing_result.get("error", "Unknown categorization error in workflow")))
                result.categorization_result = {
                    "document_type": "Error",
                    "confidence": 0.0,
                    "reasoning": error_reason
                }
                result.confidence_scores["categorization"] = 0.0
                logger.warning(f"File {file_id}: Workflow Categorization Part FAILED or Errored. Reason: {error_reason}")

            # Handle metadata part
            metadata_details_from_workflow = processing_result.get("metadata") # Can be None if not processed
            if field_names and tool_execution_successful and metadata_details_from_workflow is not None:
                if "error" in metadata_details_from_workflow:
                    tool_execution_successful = False
                    result.metadata_result = {"error": metadata_details_from_workflow["error"]}
                    logger.warning(f"File {file_id}: Metadata extraction part of workflow reported error: {metadata_details_from_workflow['error']}")
                else:
                    result.metadata_result = metadata_details_from_workflow
                    logger.info(f"File {file_id}: Agent received metadata part of workflow: {result.metadata_result}")
                    for key, value in result.metadata_result.items():
                        if key.endswith("_confidence_numeric") and isinstance(value, (int, float)):
                            meta_field_name = key.replace("_confidence_numeric", "")
                            result.confidence_scores["metadata"][meta_field_name] = value
            elif field_names and tool_execution_successful and metadata_details_from_workflow is None:
                 logger.info(f"File {file_id}: Metadata extraction returned no data from workflow, though workflow reported success.")
                 result.metadata_result = {}
            elif not field_names and tool_execution_successful:
                logger.info(f"File {file_id}: No metadata fields were requested for extraction.")
                result.metadata_result = {}
            elif not tool_execution_successful:
                 logger.info(f"File {file_id}: Metadata processing skipped due to earlier failure in workflow.")
                 result.metadata_result = {}


            # Ensure confidence_scores["metadata"] exists even if empty
            if "metadata" not in result.confidence_scores:
                result.confidence_scores["metadata"] = {}

            # Overall confidence from workflow (or could be re-calculated based on components if desired)
            result.confidence_scores["overall"] = processing_result.get("overall_confidence", 0.0)

            # If the tool execution itself failed, set status to ERROR and skip confidence checks
            if not tool_execution_successful:
                result.status = ProcessingStatus.ERROR
                # Prefer more specific error from categorization or metadata if available
                cat_error = result.categorization_result.get("reasoning", "") if result.categorization_result.get("document_type") == "Error" else ""
                meta_error = result.metadata_result.get("error", "") if result.metadata_result else ""
                error_priority = [cat_error, meta_error, "Tool execution failed or sub-task reported error."]
                result.error_message = next((err for err in error_priority if err), "Tool execution failed")
                logger.error(f"File {file_id}: Tool execution marked as FAILED. Final error: {result.error_message}")
            else:
                # Proceed with confidence boundary checks
                confidence_scores_json = json.dumps({
                    "categorization_confidence": result.confidence_scores["categorization"],
                    "metadata_confidences": result.confidence_scores.get("metadata", {})
                })
                logger.info(f"File {file_id}: Calling _check_confidence_boundaries with: {confidence_scores_json}")

                boundaries_result = json.loads(self._check_confidence_boundaries(confidence_scores_json))
                action = boundaries_result.get("action", "human_review")

                # Apply action based on confidence
                if action == "auto_approve":
                    result.status = ProcessingStatus.AUTO_APPROVED
                elif action == "human_review":
                    result.status = ProcessingStatus.HUMAN_REVIEW_REQUIRED
                    result.escalation_reason = boundaries_result.get("reasoning", "Confidence below auto-approve threshold")
                    self.human_review_queue.append(file_id)
                else:  # auto_reject or error from boundary check
                    result.status = ProcessingStatus.ERROR # Or a more specific status like AUTO_REJECTED
                    result.error_message = boundaries_result.get("reasoning", "Confidence too low for processing or boundary check error")

            # Check for additional escalation criteria (only if not already an error or human review by confidence)
            if tool_execution_successful and result.status not in [ProcessingStatus.ERROR, ProcessingStatus.HUMAN_REVIEW_REQUIRED]:
                if self.boundaries.escalate_on_category_change:
                    # Placeholder: Implement actual check if original category (if any) vs new category
                    pass
                if self.boundaries.escalate_on_field_inconsistency:
                    # Placeholder: Implement actual check for field inconsistencies
                    pass
            # This is the end of the 'else' block for 'if not tool_execution_successful:'
        # The main 'except' block for the 'try' that starts after initializing 'result'
        except Exception as e:
            logger.error(f"Error processing document {file_id}: {str(e)}")
            result.status = ProcessingStatus.ERROR
            result.error_message = str(e)
        
        # Record processing time
        result.processing_time = time.time() - start_time
        
        # Store result
        self.processing_results[file_id] = result
        
        return result
    
    def process_batch(
        self,
        file_ids: List[str],
        field_definitions: List[Dict[str, Any]],
        categories: Optional[List[str]] = None
    ) -> Dict[str, DocumentProcessingResult]:
        """Process a batch of documents."""
        
        # Limit batch size
        if len(file_ids) > self.boundaries.max_files_per_batch:
            logger.warning(f"Batch size {len(file_ids)} exceeds limit {self.boundaries.max_files_per_batch}")
            file_ids = file_ids[:self.boundaries.max_files_per_batch]
        
        results = {}
        
        for file_id in file_ids:
            result = self.process_document(
                file_id=file_id,
                field_definitions=field_definitions,
                categories=categories
            )
            results[file_id] = result
        
        return results
    
    def get_human_review_queue(self) -> List[Dict[str, Any]]:
        """Get documents waiting for human review."""
        review_items = []
        
        for file_id in self.human_review_queue:
            if file_id in self.processing_results:
                result = self.processing_results[file_id]
                review_items.append({
                    "file_id": file_id,
                    "file_name": result.file_name,
                    "escalation_reason": result.escalation_reason,
                    "categorization_result": result.categorization_result,
                    "metadata_result": result.metadata_result,
                    "confidence_scores": result.confidence_scores
                })
        
        return review_items
    
    def submit_human_feedback(
        self,
        file_id: str,
        approved: bool,
        feedback: Dict[str, Any]
    ) -> bool:
        """Submit human feedback for a document in review."""
        
        if file_id not in self.processing_results:
            logger.error(f"File {file_id} not found in processing results")
            return False
        
        result = self.processing_results[file_id]
        
        if approved:
            result.status = ProcessingStatus.HUMAN_APPROVED
        else:
            result.status = ProcessingStatus.HUMAN_REJECTED
        
        result.human_feedback = feedback
        
        # Remove from human review queue
        if file_id in self.human_review_queue:
            self.human_review_queue.remove(file_id)
        
        logger.info(f"Human feedback submitted for {file_id}: {'Approved' if approved else 'Rejected'}")
        
        return True
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of processing results."""
        
        status_counts = {}
        total_files = len(self.processing_results)
        
        for result in self.processing_results.values():
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        avg_processing_time = 0
        if total_files > 0:
            total_time = sum(
                result.processing_time for result in self.processing_results.values()
                if result.processing_time is not None
            )
            avg_processing_time = total_time / total_files
        
        return {
            "total_files": total_files,
            "status_counts": status_counts,
            "human_review_queue_size": len(self.human_review_queue),
            "average_processing_time": avg_processing_time,
            "boundaries": {
                "auto_approve_threshold": self.boundaries.auto_approve_threshold,
                "human_review_threshold": self.boundaries.human_review_threshold,
                "auto_reject_threshold": self.boundaries.auto_reject_threshold
            }
        }

class StreamlitAgentCallback(BaseCallbackHandler):
    """Callback handler for displaying agent progress in Streamlit."""
    
    def __init__(self):
        self.progress_container = None
        self.status_container = None
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent takes an action."""
        if self.status_container:
            self.status_container.info(f"Agent action: {action.tool}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes."""
        if self.status_container:
            self.status_container.success("Agent completed processing")

def create_box_ai_agent_ui():
    """Create Streamlit UI for the Box AI document processing agent."""
    
    st.title("🤖 Box AI Document Processing Agent")
    
    # Initialize agent if not exists
    if "box_ai_processing_agent" not in st.session_state:
        # Default boundaries
        boundaries = ProcessingBoundaries()
        
        # Create agent
        st.session_state.box_ai_processing_agent = BoxAIDocumentProcessingAgent(
            client=st.session_state.client,
            boundaries=boundaries,
            enable_human_loop=True
        )
    
    agent = st.session_state.box_ai_processing_agent
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Agent Control", "Processing Results", "Human Review", "Settings"])
    
    with tab1:
        st.header("Agent Control Panel")
        
        # File selection
        st.subheader("Document Selection")
        
        selection_mode = st.radio(
            "Selection Mode",
            ["Selected Files", "Box Folder"],
            help="Choose whether to process selected files or all files in a Box folder"
        )
        
        files_to_process = []
        
        if selection_mode == "Selected Files":
            if "selected_files" in st.session_state and st.session_state.selected_files:
                # Ensure selected_files contains dictionaries with an 'id' key
                valid_files = [f for f in st.session_state.selected_files if isinstance(f, dict) and 'id' in f]
                if len(valid_files) != len(st.session_state.selected_files):
                    logger.warning("Some items in st.session_state.selected_files were not in the expected format (dict with 'id').")
                files_to_process = [f['id'] for f in valid_files] # Extract only string IDs
                st.success(f"{len(files_to_process)} files selected for processing.")
                if not files_to_process and st.session_state.selected_files: # If all were invalid
                     st.error("Selected files are not in the correct format. Please re-select from File Browser.")
            else:
                st.warning("No files selected. Please select files in the File Browser tab.")
        else:
            folder_id = st.text_input("Box Folder ID", value="0")
            if folder_id:
                try:
                    folder_items = st.session_state.client.folder(folder_id=folder_id).get_items()
                    files_to_process = [item.id for item in folder_items if item.type == "file"]
                    st.success(f"Found {len(files_to_process)} files in folder")
                except Exception as e:
                    st.error(f"Error accessing folder: {str(e)}")

        # Display Document Categories from main configuration
        st.subheader("Document Categories (from main configuration)")
        configured_categories = []
        if "document_types" in st.session_state and st.session_state.document_types:
            configured_categories = [dtype["name"] for dtype in st.session_state.document_types]
            with st.expander("View categories the agent will use", expanded=False):
                for cat_name in configured_categories:
                    st.markdown(f"- {cat_name}")
            if not configured_categories:
                st.warning("No document categories found in the main configuration (Document Categorization > Settings). Please configure them first.")
        else:
            st.warning("Document categories not initialized in session state. Please visit 'Document Categorization' page first.")
            
        categories_to_pass = configured_categories

        st.subheader("Metadata Fields (from main 'Metadata Configuration')")
        metadata_config = st.session_state.get("metadata_config", {})
        field_definitions_to_pass = [] # Initialize

        if metadata_config.get("use_template") and metadata_config.get("template_id"):
            template_id = metadata_config["template_id"]
            st.info(f"Agent will use fields from the currently selected template: **{template_id}** (defined in 'Metadata Configuration').")

            # Attempt to get fields from the template
            try:
                # Parse template_id to scope and template_key
                if template_id.startswith('enterprise_'):
                    parts = template_id.split('_', 2)
                    scope = 'enterprise'
                    template_key = parts[2] if len(parts) >= 3 else template_id
                else:
                    scope = 'enterprise'
                    template_key = template_id
                
                if "client" in st.session_state and st.session_state.client:
                    retrieved_fields = get_fields_for_ai_from_template(scope, template_key)
                    if retrieved_fields:
                        field_definitions_to_pass = retrieved_fields
                        with st.expander("View fields from selected template", expanded=False):
                            for field_def in field_definitions_to_pass:
                                st.markdown(f"- **{field_def.get('displayName', field_def.get('key'))}** (Type: {field_def.get('type')})")
                    else:
                        st.error(f"Could not retrieve fields for template '{template_id}'. Agent will proceed without specific fields for extraction.")
                else:
                    st.error("Box client not available. Cannot retrieve template fields. Agent will proceed without specific fields.")
            except Exception as e:
                st.error(f"Error retrieving fields for template '{template_id}': {e}")
                logger.error(f"Error in agent UI getting template fields for {template_id}: {e}")

        elif metadata_config.get("extraction_method") == "freeform":
            st.info("Agent will operate in 'freeform' extraction mode based on main 'Metadata Configuration'. No specific template fields will be targeted.")
            # field_definitions_to_pass remains []
        
        else:
            st.warning("No metadata template selected in 'Metadata Configuration', and not set to 'freeform'. Agent will primarily focus on categorization and may not extract specific metadata fields.")
            # field_definitions_to_pass remains []

        # Start processing
        if st.button("🚀 Start Autonomous Processing", type="primary"):
            if not files_to_process:
                st.error("No files selected for processing")
            # Note: We allow processing even if field_definitions_to_pass is empty,
            # as the agent might still perform categorization or very basic freeform extraction.
            # elif not field_definitions_to_pass:
            #     st.error("No field definitions derived from configuration.")
            else:
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process files
                with st.spinner("Agent is processing documents with Box AI..."):
                    results = agent.process_batch(
                        file_ids=files_to_process,
                        field_definitions=field_definitions_to_pass,
                        categories=categories_to_pass
                    )
                
                progress_bar.progress(1.0)
                status_text.success(f"Processed {len(results)} files")
                
                # Display summary
                summary = agent.get_processing_summary()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", summary["total_files"])
                with col2:
                    st.metric("Auto Approved", summary["status_counts"].get("auto_approved", 0))
                with col3:
                    st.metric("Human Review Required", summary["human_review_queue_size"])
    
    with tab2:
        st.header("Processing Results")
        
        # Get processing summary
        summary = agent.get_processing_summary()
        
        if summary["total_files"] > 0:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Files", summary["total_files"])
            with col2:
                st.metric("Auto Approved", summary["status_counts"].get("auto_approved", 0))
            with col3:
                st.metric("Human Review", summary["human_review_queue_size"])
            with col4:
                st.metric("Errors", summary["status_counts"].get("error", 0))
            
            # Display detailed results
            st.subheader("Detailed Results")
            
            results_data = []
            for file_id, result in agent.processing_results.items():
                results_data.append({
                    "File ID": file_id,
                    "File Name": result.file_name,
                    "Status": result.status.value,
                    "Category": result.categorization_result.get("document_type", "Unknown") if result.categorization_result else "Unknown",
                    "Confidence": result.confidence_scores.get("overall", 0.0) if result.confidence_scores else 0.0,
                    "Processing Time": f"{result.processing_time:.2f}s" if result.processing_time else "N/A"
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Detailed view for selected file
            st.subheader("Detailed View")
            
            selected_file_id = st.selectbox(
                "Select File for Detailed View",
                options=list(agent.processing_results.keys()),
                format_func=lambda x: agent.processing_results[x].file_name
            )
            
            if selected_file_id:
                result = agent.processing_results[selected_file_id]
                
                # Display basic information
                st.markdown(f"**File Name:** {result.file_name}")
                st.markdown(f"**Status:** {result.status.value}")
                
                # Display categorization result
                if result.categorization_result:
                    st.subheader("Categorization")
                    st.markdown(f"**Category:** {result.categorization_result.get('document_type', 'Unknown')}")
                    st.markdown(f"**Confidence:** {result.categorization_result.get('confidence', 0.0):.2f}")
                    st.markdown(f"**Reasoning:** {result.categorization_result.get('reasoning', 'No reasoning provided')}")
                
                # Display metadata result
                if result.metadata_result:
                    st.subheader("Metadata")
                    
                    metadata_rows = []
                    for key, value in result.metadata_result.items():
                        if not key.endswith("_confidence") and not key.endswith("_confidence_numeric"):
                            confidence_key = f"{key}_confidence"
                            confidence = result.metadata_result.get(confidence_key, "Unknown")
                            
                            metadata_rows.append({
                                "Field": key,
                                "Value": value,
                                "Confidence": confidence
                            })
                    
                    if metadata_rows:
                        metadata_df = pd.DataFrame(metadata_rows)
                        st.dataframe(metadata_df, use_container_width=True)
                
                # Display confidence scores
                if result.confidence_scores:
                    st.subheader("Confidence Scores")
                    
                    # Overall confidence
                    overall = result.confidence_scores.get("overall", 0.0)
                    
                    # Determine color based on confidence
                    if overall >= agent.boundaries.auto_approve_threshold:
                        color = "#66BB6A"  # Green
                    elif overall >= agent.boundaries.human_review_threshold:
                        color = "#FFA726"  # Orange
                    else:
                        color = "#FF5252"  # Red
                    
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {color};
                            padding: 10px;
                            border-radius: 5px;
                            text-align: center;
                            color: white;
                            font-weight: bold;
                            font-size: 24px;
                            margin-bottom: 10px;
                        ">
                        Overall Confidence: {overall:.2f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Display confidence breakdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Categorization Confidence:** {result.confidence_scores.get('categorization', 0.0):.2f}")
                    
                    with col2:
                        metadata_confidences = result.confidence_scores.get("metadata", {})
                        if metadata_confidences:
                            avg_metadata = sum(metadata_confidences.values()) / len(metadata_confidences)
                            st.markdown(f"**Average Metadata Confidence:** {avg_metadata:.2f}")
                
                # Display error message if any
                if result.error_message:
                    st.error(f"Error: {result.error_message}")
                
                # Display escalation reason if any
                if result.escalation_reason:
                    st.warning(f"Escalation Reason: {result.escalation_reason}")
                
                # Display human feedback if any
                if result.human_feedback:
                    st.subheader("Human Feedback")
                    st.markdown(f"**Reviewer:** {result.human_feedback.get('reviewer', 'Unknown')}")
                    st.markdown(f"**Timestamp:** {result.human_feedback.get('timestamp', 'Unknown')}")
                    if "comments" in result.human_feedback:
                        st.markdown(f"**Comments:** {result.human_feedback['comments']}")
        else:
            st.info("No processing results yet. Start processing in the Agent Control tab.")
    
    with tab3:
        st.header("Human Review Queue")
        
        review_items = agent.get_human_review_queue()
        
        if review_items:
            st.warning(f"{len(review_items)} documents require human review")
            
            for item in review_items:
                with st.expander(f"Review: {item['file_name']}"):
                    st.markdown(f"**Escalation Reason:** {item['escalation_reason']}")
                    
                    # Display categorization result
                    if item['categorization_result']:
                        st.subheader("Categorization")
                        cat_result = item['categorization_result']
                        st.markdown(f"**Category:** {cat_result.get('document_type', 'Unknown')}")
                        st.markdown(f"**Confidence:** {cat_result.get('confidence', 0.0):.2f}")
                        st.markdown(f"**Reasoning:** {cat_result.get('reasoning', 'No reasoning provided')}")
                    
                    # Display metadata result
                    if item['metadata_result']:
                        st.subheader("Metadata")
                        # Display metadata fields
                        for field_name, value in item['metadata_result'].items():
                            if not field_name.endswith('_confidence') and not field_name.endswith('_confidence_numeric'):
                                confidence_key = f"{field_name}_confidence"
                                confidence = item['metadata_result'].get(confidence_key, "Unknown")
                                st.markdown(f"**{field_name}:** {value} (Confidence: {confidence})")
                    
                    # Human decision
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"✅ Approve", key=f"approve_{item['file_id']}"):
                            agent.submit_human_feedback(
                                file_id=item['file_id'],
                                approved=True,
                                feedback={"reviewer": "human", "timestamp": datetime.now().isoformat()}
                            )
                            st.success("Document approved!")
                            st.rerun()
                    
                    with col2:
                        if st.button(f"❌ Reject", key=f"reject_{item['file_id']}"):
                            agent.submit_human_feedback(
                                file_id=item['file_id'],
                                approved=False,
                                feedback={"reviewer": "human", "timestamp": datetime.now().isoformat()}
                            )
                            st.error("Document rejected!")
                            st.rerun()
        else:
            st.success("No documents in human review queue")
    
    with tab4:
        st.header("Agent Settings")
        
        # Processing boundaries
        st.subheader("Processing Boundaries")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auto_approve = st.slider(
                "Auto-Approve Threshold",
                min_value=0.0,
                max_value=1.0,
                value=agent.boundaries.auto_approve_threshold,
                step=0.05,
                help="Documents with confidence above this threshold are automatically approved"
            )
        
        with col2:
            human_review = st.slider(
                "Human Review Threshold",
                min_value=0.0,
                max_value=1.0,
                value=agent.boundaries.human_review_threshold,
                step=0.05,
                help="Documents with confidence above this threshold require human review"
            )
        
        with col3:
            auto_reject = st.slider(
                "Auto-Reject Threshold",
                min_value=0.0,
                max_value=1.0,
                value=agent.boundaries.auto_reject_threshold,
                step=0.05,
                help="Documents with confidence below this threshold are automatically rejected"
            )
        
        # Update boundaries if changed
        if (auto_approve != agent.boundaries.auto_approve_threshold or
            human_review != agent.boundaries.human_review_threshold or
            auto_reject != agent.boundaries.auto_reject_threshold):
            
            agent.boundaries.auto_approve_threshold = auto_approve
            agent.boundaries.human_review_threshold = human_review
            agent.boundaries.auto_reject_threshold = auto_reject
            
            st.success("Processing boundaries updated")
        
        # Escalation settings
        st.subheader("Escalation Settings")
        
        escalate_category_change = st.checkbox(
            "Escalate on Category Change",
            value=agent.boundaries.escalate_on_category_change,
            help="Escalate to human review if category changes during processing"
        )
        
        escalate_low_metadata = st.checkbox(
            "Escalate on Low Metadata Confidence",
            value=agent.boundaries.escalate_on_low_metadata_confidence,
            help="Escalate to human review if metadata confidence is low"
        )
        
        escalate_field_inconsistency = st.checkbox(
            "Escalate on Field Inconsistency",
            value=agent.boundaries.escalate_on_field_inconsistency,
            help="Escalate to human review if field values are inconsistent"
        )
        
        # Update escalation settings
        agent.boundaries.escalate_on_category_change = escalate_category_change
        agent.boundaries.escalate_on_low_metadata_confidence = escalate_low_metadata
        agent.boundaries.escalate_on_field_inconsistency = escalate_field_inconsistency
        
        # Processing limits
        st.subheader("Processing Limits")
        
        max_files = st.number_input(
            "Max Files per Batch",
            min_value=1,
            max_value=100,
            value=agent.boundaries.max_files_per_batch,
            help="Maximum number of files to process in a single batch"
        )
        
        max_time = st.number_input(
            "Max Processing Time (minutes)",
            min_value=1,
            max_value=120,
            value=agent.boundaries.max_processing_time_minutes,
            help="Maximum time to spend processing a single batch"
        )
        
        # Update processing limits
        agent.boundaries.max_files_per_batch = max_files
        agent.boundaries.max_processing_time_minutes = max_time
