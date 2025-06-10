"""
Integration module for Box AI agent with Streamlit UI.

This module provides functions to integrate the Box AI agent with the
Streamlit UI, including tab creation, navigation, and UI components.
"""

import streamlit as st
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import Box AI agent components
from modules.box_ai_agent import (
    BoxAIDocumentProcessingAgent,
    ProcessingBoundaries,
    ProcessingStatus,
    DocumentProcessingResult,
    create_box_ai_agent_ui
)

logger = logging.getLogger(__name__)

def integrate_box_ai_agent_with_app():
    """
    Integrate Box AI agent with the Streamlit app.
    
    This function initializes the Box AI agent and adds it to the session state,
    and updates the app tabs to include the Box AI Agent tab.
    """
    # Check if client exists in session state
    if 'client' not in st.session_state:
        st.error("Box client not initialized. Please initialize the client first.")
        return
    
    # Initialize Box AI agent if not exists
    if 'box_ai_processing_agent' not in st.session_state:
        # Default boundaries
        boundaries = ProcessingBoundaries()

        # Attempt to load OpenAI API key from Streamlit secrets
        openai_api_key_value = None # Initialize to ensure it's defined
        if hasattr(st, 'secrets'):
            logger.info(f"st.secrets type: {type(st.secrets)}") # Log type of st.secrets

            box_dev_section = st.secrets.get("box_dev") # Use .get() for safety
            if box_dev_section is not None:
                logger.info(f"st.secrets['box_dev'] type: {type(box_dev_section)}")
                if hasattr(box_dev_section, 'get'): # Check if it's dict-like (supports .get())
                    openai_api_key_value = box_dev_section.get("OPENAI_API_KEY")
                elif isinstance(box_dev_section, dict): # Fallback for plain dict
                     openai_api_key_value = box_dev_section.get("OPENAI_API_KEY")

                if openai_api_key_value:
                    logger.info(f"Attempted to load OPENAI_API_KEY from st.secrets['box_dev']. Value type: {type(openai_api_key_value)}")

            # Fallback to top-level key if not found or box_dev_section was None or key not in box_dev_section
            if not openai_api_key_value:
                if hasattr(st.secrets, 'get'):
                    openai_api_key_value = st.secrets.get("OPENAI_API_KEY")
                    if openai_api_key_value:
                        logger.info(f"Attempted to load OPENAI_API_KEY directly from st.secrets. Value type: {type(openai_api_key_value)}")

            if not openai_api_key_value:
                logger.warning("OPENAI_API_KEY not found in st.secrets (neither under 'box_dev' nor at top level).")
            elif not isinstance(openai_api_key_value, str):
                # If what we got is not a string, log an error and force it to None.
                # This is crucial to prevent passing a dict/AttrDict to ChatOpenAI.
                logger.error(f"OPENAI_API_KEY found but is not a string. Type: {type(openai_api_key_value)}. Actual value: {str(openai_api_key_value)[:100]}...") # Log first 100 chars
                openai_api_key_value = None

        else:
            logger.warning("st.secrets attribute not available. Cannot load OPENAI_API_KEY from secrets.")

        openai_api_key = openai_api_key_value # This variable is then passed to BoxAIDocumentProcessingAgent

        # Create agent
        st.session_state.box_ai_processing_agent = BoxAIDocumentProcessingAgent(
            client=st.session_state.client,
            boundaries=boundaries,
            enable_human_loop=True,
            openai_api_key=openai_api_key # Pass the loaded key
        )
    
    # Update tabs if needed
    if 'tabs' in st.session_state:
        if "Box AI Agent" not in st.session_state.tabs:
            st.session_state.tabs.append("Box AI Agent")

def box_ai_agent_workflow_tab():
    """
    Display the Box AI Agent workflow tab in the Streamlit UI.
    
    This function creates the UI for the Box AI Agent tab, including
    agent control panel, processing results, human review interface,
    and settings.
    """
    # Check if agent exists in session state
    if 'box_ai_processing_agent' not in st.session_state:
        st.error("Box AI agent not initialized. Please initialize the agent first.")
        return
    
    # Create Box AI agent UI
    create_box_ai_agent_ui()

def get_human_review_count() -> int:
    """
    Get the number of documents waiting for human review.
    
    Returns:
        Number of documents in human review queue
    """
    # Check if agent exists in session state
    if 'box_ai_processing_agent' not in st.session_state:
        return 0
    
    # Get human review queue
    agent = st.session_state.box_ai_processing_agent
    return len(agent.human_review_queue)

def add_box_ai_agent_to_sidebar():
    """
    Add Box AI agent information and controls to the sidebar.
    
    This function adds human review notifications and quick actions
    to the Streamlit sidebar.
    """
    # Check if agent exists in session state
    if 'box_ai_processing_agent' not in st.session_state:
        return
    
    # Get human review count
    human_review_count = get_human_review_count()
    
    # Add to sidebar if there are documents to review
    if human_review_count > 0:
        with st.sidebar:
            st.warning(f"📋 {human_review_count} documents need human review")
            
            if st.button("Go to Human Review"):
                # Set selected tab to Box AI Agent
                if 'selected_tab' in st.session_state:
                    st.session_state.selected_tab = "Box AI Agent"
                
                # Set query params to navigate to the tab
                st.experimental_set_query_params(active_tab="Box AI Agent")

def update_app_for_box_ai_agent():
    """
    Update the main app.py file to include Box AI agent integration.
    
    This function should be called from app.py to integrate the Box AI agent
    with the Streamlit app.
    """
    # Initialize Box AI agent
    integrate_box_ai_agent_with_app()
    
    # Add Box AI agent to sidebar
    add_box_ai_agent_to_sidebar()
    
    # Handle tab selection
    if 'selected_tab' in st.session_state and st.session_state.selected_tab == "Box AI Agent":
        box_ai_agent_workflow_tab()
