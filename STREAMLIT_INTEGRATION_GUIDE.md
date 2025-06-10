# Box AI Agent Streamlit Integration Guide

## Overview

This guide provides comprehensive instructions for integrating the Box AI agent with your Streamlit application. The integration enables autonomous document processing with human-in-the-loop capabilities directly within your existing Streamlit UI.

## Table of Contents

1. [Integration Steps](#integration-steps)
2. [UI Components](#ui-components)
3. [Workflow Guide](#workflow-guide)
4. [Customization Options](#customization-options)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Configuration](#advanced-configuration)

## Integration Steps

Follow these steps to integrate the Box AI agent with your Streamlit application:

### 1. Add Required Imports

Add the following imports to your `app.py` file:

```python
# Import Box AI integration components
from modules.box_ai_streamlit_integration import (
    update_app_for_box_ai_agent,
    box_ai_agent_workflow_tab
)
```

### 2. Initialize the Integration

Add the following code to your main function in `app.py`:

```python
def main():
    # Your existing code...
    
    # Add Box AI agent integration
    update_app_for_box_ai_agent()
    
    # Handle tab selection
    if st.session_state.selected_tab == "Box AI Agent":
        box_ai_agent_workflow_tab()
```

### 3. Update Tab Navigation

If you have custom tab navigation, ensure the Box AI Agent tab is included:

```python
# Example tab navigation code
tabs = ["File Browser", "Document Categorization", "Metadata Extraction", "Box AI Agent"]
selected_tab = st.sidebar.radio("Navigation", tabs)
st.session_state.selected_tab = selected_tab
```

### 4. Run the Application

Start your Streamlit application as usual:

```bash
streamlit run app.py
```

## UI Components

The Box AI agent integration adds the following UI components to your Streamlit application:

### 1. Box AI Agent Tab

A new tab in your Streamlit interface with four sub-sections:

- **Agent Control Panel**: Configure and start autonomous processing
- **Processing Results**: View detailed results with confidence visualizations
- **Human Review Interface**: Review and approve/reject documents requiring human verification
- **Settings Panel**: Adjust confidence thresholds and agent behavior

### 2. Sidebar Notifications

- Human review notifications showing the number of documents waiting for review
- Quick navigation button to the human review interface

### 3. Agent Control Panel

The agent control panel includes:

- **Document Selection**: Select files for processing
- **Category Configuration**: Define document categories
- **Field Definition**: Configure metadata fields to extract
- **Processing Controls**: Start autonomous processing and monitor progress

### 4. Results Dashboard

The results dashboard displays:

- **Summary Metrics**: Total files processed, auto-approved, human review required, etc.
- **Detailed Results Table**: List of all processed documents with status and confidence
- **Detailed View**: In-depth view of categorization and metadata results for selected documents
- **Confidence Visualizations**: Visual representation of confidence scores with color coding

### 5. Human Review Interface

The human review interface shows:

- **Review Queue**: List of documents requiring human review
- **Document Details**: Categorization and metadata results with confidence scores
- **Escalation Reason**: Explanation of why the document was escalated
- **Approval Controls**: Buttons to approve or reject the document

### 6. Settings Panel

The settings panel allows configuration of:

- **Confidence Thresholds**: Auto-approve, human review, and auto-reject thresholds
- **Escalation Criteria**: Configure when documents should be escalated to human review
- **Processing Limits**: Set batch size and processing time limits

## Workflow Guide

### Autonomous Processing Workflow

1. Navigate to the "Box AI Agent" tab
2. Select documents for processing
3. Configure categories and fields (or use defaults)
4. Click "Start Autonomous Processing"
5. Monitor progress in the status area
6. View results in the Processing Results section

### Human Review Workflow

1. When documents require human review, a notification appears in the sidebar
2. Click "Go to Human Review" or navigate to the Human Review section
3. Review each document's details and confidence scores
4. Approve or reject the document
5. Add optional feedback comments
6. The document is removed from the review queue after decision

### Settings Adjustment Workflow

1. Navigate to the Settings section in the Box AI Agent tab
2. Adjust confidence thresholds using sliders
3. Toggle escalation criteria on/off
4. Set processing limits
5. Changes take effect immediately for future processing

## Customization Options

### Confidence Thresholds

Adjust confidence thresholds to control the agent's decision-making:

```python
from modules.box_ai_agent import ProcessingBoundaries

# Create custom boundaries
boundaries = ProcessingBoundaries(
    auto_approve_threshold=0.90,  # Higher threshold for stricter auto-approval
    human_review_threshold=0.70,  # Higher threshold for human review
    auto_reject_threshold=0.40    # Higher threshold for auto-rejection
)

# Update agent boundaries
st.session_state.box_ai_processing_agent.boundaries = boundaries
```

### Custom Categories

Define custom document categories:

```python
# Define custom categories
custom_categories = [
    "Invoice",
    "Purchase Order",
    "Receipt",
    "Contract",
    "Legal Document",
    "HR Document",
    "Other"
]

# Use custom categories in agent processing
result = st.session_state.box_ai_processing_agent.process_document(
    file_id=file_id,
    field_definitions=field_definitions,
    categories=custom_categories
)
```

### Custom Field Definitions

Define custom metadata fields:

```python
# Define custom field definitions
custom_fields = [
    {"name": "invoice_number", "type": "string", "description": "Invoice identifier"},
    {"name": "invoice_date", "type": "date", "description": "Date invoice was issued"},
    {"name": "due_date", "type": "date", "description": "Date payment is due"},
    {"name": "vendor_name", "type": "string", "description": "Name of vendor"},
    {"name": "total_amount", "type": "number", "description": "Total invoice amount"}
]

# Use custom fields in agent processing
result = st.session_state.box_ai_processing_agent.process_document(
    file_id=file_id,
    field_definitions=custom_fields,
    categories=categories
)
```

## Troubleshooting

### Common Issues

1. **Agent Not Initializing**:
   - Check that Box client is properly initialized in session state
   - Verify that all required modules are imported
   - Check for errors in the console log

2. **UI Components Not Displaying**:
   - Verify that the Box AI Agent tab is added to your tabs list
   - Check that `box_ai_agent_workflow_tab()` is called when the tab is selected
   - Ensure Streamlit version is compatible (1.10.0 or higher recommended)

3. **Processing Errors**:
   - Check Box API authentication and permissions
   - Verify that selected files exist and are accessible
   - Check for timeout or rate limit errors in the logs

4. **Human Review Not Working**:
   - Verify that confidence thresholds are properly set
   - Check that the human review queue is being updated
   - Ensure the submit_human_feedback function is being called

### Debugging Tips

1. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Session State**:
   ```python
   st.write(st.session_state)
   ```

3. **Test Agent Directly**:
   ```python
   # Test agent processing directly
   result = st.session_state.box_ai_processing_agent.process_document(
       file_id="test_file_id",
       field_definitions=[{"name": "test", "type": "string"}],
       categories=["Test"]
   )
   st.write(result)
   ```

## Advanced Configuration

### Custom UI Layout

You can customize the UI layout by modifying the `create_box_ai_agent_ui` function:

```python
# Import the function
from modules.box_ai_agent import create_box_ai_agent_ui

# Create a custom UI function
def custom_box_ai_agent_ui():
    st.title("Custom Box AI Agent")
    
    # Your custom UI code here
    # ...
    
    # Call the original UI function with custom parameters
    create_box_ai_agent_ui(
        show_settings=True,
        show_human_review=True,
        show_results=True
    )

# Use your custom UI function
if st.session_state.selected_tab == "Box AI Agent":
    custom_box_ai_agent_ui()
```

### Integration with Other Components

You can integrate the Box AI agent with other components of your application:

```python
# Example: Use Box AI agent results in another part of your app
if "box_ai_processing_agent" in st.session_state:
    agent = st.session_state.box_ai_processing_agent
    
    # Get processing results
    results = agent.processing_results
    
    # Use results in another component
    if results:
        # Do something with the results
        pass
```

### Custom Processing Workflow

You can create a custom processing workflow:

```python
# Example: Custom processing workflow
def custom_processing_workflow(file_ids, categories, fields):
    if "box_ai_processing_agent" in st.session_state:
        agent = st.session_state.box_ai_processing_agent
        
        # Process files
        results = agent.process_batch(
            file_ids=file_ids,
            field_definitions=fields,
            categories=categories
        )
        
        # Custom post-processing
        for file_id, result in results.items():
            if result.status == ProcessingStatus.AUTO_APPROVED:
                # Do something with auto-approved documents
                pass
            elif result.status == ProcessingStatus.HUMAN_REVIEW_REQUIRED:
                # Do something with documents requiring review
                pass
        
        return results
    
    return {}
```

This guide provides a comprehensive overview of integrating the Box AI agent with your Streamlit application. For more detailed information, refer to the Box AI Agent Documentation and the API reference for each module.
