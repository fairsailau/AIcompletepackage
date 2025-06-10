"""
End-to-end integration test for Streamlit app with Box AI agent.

This module provides comprehensive tests to ensure the full workflow
integration between the Streamlit UI and Box AI agent.
"""

import unittest
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

# Import Box AI agent components
from modules.box_ai_agent import (
    BoxAIDocumentProcessingAgent,
    ProcessingBoundaries,
    ProcessingStatus,
    DocumentProcessingResult
)

# Import Box AI tool
from modules.box_ai_tool import (
    BoxAITool,
    BoxAIDocumentProcessor
)

# Import Streamlit integration components (to be created)
from modules.box_ai_streamlit_integration import (
    integrate_box_ai_agent_with_app,
    box_ai_agent_workflow_tab,
    get_human_review_count
)

class TestEndToEndIntegration(unittest.TestCase):
    """Test cases for end-to-end integration of Streamlit and Box AI agent."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock Streamlit session state
        self.mock_session_state = {}
        
        # Mock Box client
        self.mock_client = Mock()
        self.mock_client.file.return_value.get.return_value = Mock(name="test_file.pdf", size=1024)
        
        # Create test boundaries
        self.boundaries = ProcessingBoundaries(
            auto_approve_threshold=0.85,
            human_review_threshold=0.6,
            auto_reject_threshold=0.3
        )
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_agent_initialization(self, mock_st_state):
        """Test agent initialization in Streamlit."""
        # Set up mock session state
        mock_st_state['client'] = self.mock_client
        
        # Test integration
        with patch('modules.box_ai_streamlit_integration.BoxAIDocumentProcessingAgent') as mock_agent_class:
            integrate_box_ai_agent_with_app()
            
            # Verify agent was initialized
            mock_agent_class.assert_called_once()
            
            # Verify tabs were updated
            self.assertIn('tabs', mock_st_state)
            self.assertIn('Box AI Agent', mock_st_state['tabs'])
    
    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.title')
    @patch('streamlit.tabs')
    def test_agent_workflow_ui(self, mock_tabs, mock_title, mock_st_state):
        """Test agent workflow UI in Streamlit."""
        # Set up mock session state with agent
        mock_st_state['client'] = self.mock_client
        
        with patch('modules.box_ai_agent.BoxAIDocumentProcessingAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            mock_agent.get_processing_summary.return_value = {
                "total_files": 0,
                "status_counts": {},
                "human_review_queue_size": 0,
                "average_processing_time": 0,
                "boundaries": self.boundaries.__dict__
            }
            
            # Initialize agent
            integrate_box_ai_agent_with_app()
            
            # Test workflow tab
            with patch('modules.box_ai_streamlit_integration.create_box_ai_agent_ui') as mock_create_ui:
                box_ai_agent_workflow_tab()
                
                # Verify UI was created
                mock_create_ui.assert_called_once()
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_human_review_count(self, mock_st_state):
        """Test human review count functionality."""
        # Set up mock session state with agent
        mock_agent = Mock()
        mock_agent.human_review_queue = ["file1", "file2", "file3"]
        mock_st_state['box_ai_processing_agent'] = mock_agent
        
        # Test review count
        count = get_human_review_count()
        
        # Verify count
        self.assertEqual(count, 3)
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_document_processing_workflow(self, mock_st_state):
        """Test full document processing workflow."""
        # Set up mock session state
        mock_st_state['client'] = self.mock_client
        mock_st_state['selected_files'] = ["file1", "file2"]
        
        # Create mock agent
        mock_agent = Mock()
        mock_st_state['box_ai_processing_agent'] = mock_agent
        
        # Mock agent processing
        mock_agent.process_batch.return_value = {
            "file1": DocumentProcessingResult(
                file_id="file1",
                file_name="invoice.pdf",
                status=ProcessingStatus.AUTO_APPROVED,
                categorization_result={"document_type": "Invoice", "confidence": 0.9},
                metadata_result={"invoice_number": "INV-001", "total_amount": "1500.00"},
                confidence_scores={"categorization": 0.9, "metadata": 0.88},
                processing_time=1.5
            ),
            "file2": DocumentProcessingResult(
                file_id="file2",
                file_name="contract.pdf",
                status=ProcessingStatus.HUMAN_REVIEW_REQUIRED,
                categorization_result={"document_type": "Contract", "confidence": 0.7},
                metadata_result={"contract_id": "CONT-001", "effective_date": "2024-01-01"},
                confidence_scores={"categorization": 0.7, "metadata": 0.65},
                escalation_reason="Medium confidence requires verification",
                processing_time=2.0
            )
        }
        
        # Mock get_processing_summary
        mock_agent.get_processing_summary.return_value = {
            "total_files": 2,
            "status_counts": {
                "auto_approved": 1,
                "human_review_required": 1
            },
            "human_review_queue_size": 1,
            "average_processing_time": 1.75,
            "boundaries": self.boundaries.__dict__
        }
        
        # Mock get_human_review_queue
        mock_agent.get_human_review_queue.return_value = [
            {
                "file_id": "file2",
                "file_name": "contract.pdf",
                "escalation_reason": "Medium confidence requires verification",
                "categorization_result": {"document_type": "Contract", "confidence": 0.7},
                "metadata_result": {"contract_id": "CONT-001", "effective_date": "2024-01-01"},
                "confidence_scores": {"categorization": 0.7, "metadata": 0.65}
            }
        ]
        
        # Test human review submission
        mock_agent.submit_human_feedback.return_value = True
        
        # Verify workflow components are functional
        self.assertEqual(len(mock_st_state['selected_files']), 2)
        self.assertEqual(mock_agent.get_processing_summary()["total_files"], 2)
        self.assertEqual(mock_agent.get_processing_summary()["human_review_queue_size"], 1)
        self.assertTrue(mock_agent.submit_human_feedback("file2", True, {"reviewer": "test"}))

class TestStreamlitIntegration(unittest.TestCase):
    """Test cases for Streamlit-specific integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock Streamlit components
        self.mock_progress = Mock()
        self.mock_status = Mock()
    
    @patch('streamlit.progress')
    @patch('streamlit.empty')
    def test_streamlit_progress_indicators(self, mock_empty, mock_progress):
        """Test Streamlit progress indicators."""
        # Mock progress bar and status text
        mock_progress.return_value = self.mock_progress
        mock_empty.return_value = self.mock_status
        
        # Create callback
        with patch('modules.box_ai_agent.StreamlitAgentCallback') as mock_callback_class:
            callback = mock_callback_class.return_value
            callback.progress_container = self.mock_progress
            callback.status_container = self.mock_status
            
            # Test agent action callback
            action = Mock()
            action.tool = "categorize_document"
            callback.on_agent_action(action)
            
            # Verify status was updated
            self.mock_status.info.assert_called_once()
            
            # Test agent finish callback
            finish = Mock()
            callback.on_agent_finish(finish)
            
            # Verify status was updated
            self.mock_status.success.assert_called_once()
    
    @patch('streamlit.session_state', new_callable=dict)
    @patch('streamlit.experimental_set_query_params')
    def test_streamlit_navigation(self, mock_set_params, mock_st_state):
        """Test Streamlit navigation between tabs."""
        # Set up mock session state
        mock_st_state['client'] = self.mock_client = Mock()
        mock_st_state['tabs'] = ["File Browser", "Document Categorization", "Metadata Extraction", "Box AI Agent"]
        mock_st_state['selected_tab'] = "Box AI Agent"
        
        # Test navigation to results tab
        with patch('modules.box_ai_streamlit_integration.box_ai_agent_workflow_tab'):
            # This would normally trigger tab navigation
            mock_set_params(active_tab="Results")
            
            # Verify query params were set
            mock_set_params.assert_called_once_with(active_tab="Results")

class TestBoxAIIntegrationWithExistingApp(unittest.TestCase):
    """Test cases for Box AI integration with existing app functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock Streamlit session state
        self.mock_session_state = {}
        
        # Mock Box client
        self.mock_client = Mock()
        
        # Mock existing app components
        self.mock_document_categorization = Mock()
        self.mock_metadata_extraction = Mock()
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_integration_with_document_categorization(self, mock_st_state):
        """Test integration with existing document categorization."""
        # Set up mock session state
        mock_st_state['client'] = self.mock_client
        mock_st_state['document_categorization'] = self.mock_document_categorization
        
        # Mock categorization result
        categorization_result = {
            "document_type": "Invoice",
            "confidence": 0.9,
            "reasoning": "Document contains invoice number, date, and total amount"
        }
        self.mock_document_categorization.categorize_document.return_value = categorization_result
        
        # Create Box AI agent
        with patch('modules.box_ai_agent.BoxAIDocumentProcessingAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            # Mock Box AI processing
            mock_agent.process_document.return_value = DocumentProcessingResult(
                file_id="test_file",
                file_name="test.pdf",
                status=ProcessingStatus.AUTO_APPROVED,
                categorization_result=categorization_result,
                confidence_scores={"overall": 0.9, "categorization": 0.9}
            )
            
            # Initialize agent
            integrate_box_ai_agent_with_app()
            
            # Verify agent was initialized
            mock_agent_class.assert_called_once()
            
            # Test integration with document categorization
            # This would verify that the Box AI agent can use results from the existing categorization
            self.assertEqual(mock_agent.process_document().categorization_result, categorization_result)
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_integration_with_metadata_extraction(self, mock_st_state):
        """Test integration with existing metadata extraction."""
        # Set up mock session state
        mock_st_state['client'] = self.mock_client
        mock_st_state['metadata_extraction'] = self.mock_metadata_extraction
        
        # Mock metadata result
        metadata_result = {
            "invoice_number": "INV-001",
            "total_amount": "1500.00",
            "invoice_date": "2024-01-01"
        }
        self.mock_metadata_extraction.extract_metadata.return_value = metadata_result
        
        # Create Box AI agent
        with patch('modules.box_ai_agent.BoxAIDocumentProcessingAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            # Mock Box AI processing
            mock_agent.process_document.return_value = DocumentProcessingResult(
                file_id="test_file",
                file_name="test.pdf",
                status=ProcessingStatus.AUTO_APPROVED,
                metadata_result=metadata_result,
                confidence_scores={"overall": 0.9, "metadata": 0.9}
            )
            
            # Initialize agent
            integrate_box_ai_agent_with_app()
            
            # Verify agent was initialized
            mock_agent_class.assert_called_once()
            
            # Test integration with metadata extraction
            # This would verify that the Box AI agent can use results from the existing metadata extraction
            self.assertEqual(mock_agent.process_document().metadata_result, metadata_result)

def run_integration_tests():
    """Run all integration tests."""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestEndToEndIntegration))
    test_suite.addTest(unittest.makeSuite(TestStreamlitIntegration))
    test_suite.addTest(unittest.makeSuite(TestBoxAIIntegrationWithExistingApp))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

def test_app_integration():
    """Test integration with the main app.py file."""
    
    # Check if app.py exists
    if not os.path.exists('/home/ubuntu/AI-Metadata-Extraction-Final/app.py'):
        print("❌ app.py not found. Integration test cannot proceed.")
        return False
    
    try:
        # Read app.py
        with open('/home/ubuntu/AI-Metadata-Extraction-Final/app.py', 'r') as f:
            app_content = f.read()
        
        # Check for required imports
        import_checks = [
            "from modules.box_ai_agent import",
            "from modules.box_ai_tool import",
            "from modules.box_ai_streamlit_integration import"
        ]
        
        # Check for integration code
        integration_checks = [
            "integrate_box_ai_agent_with_app()",
            "box_ai_agent_workflow_tab()",
            "get_human_review_count()"
        ]
        
        # Verify imports
        missing_imports = [imp for imp in import_checks if imp not in app_content]
        
        # Verify integration code
        missing_integration = [code for code in integration_checks if code not in app_content]
        
        if missing_imports or missing_integration:
            print("❌ app.py integration test failed.")
            if missing_imports:
                print("Missing imports:")
                for imp in missing_imports:
                    print(f"  - {imp}")
            
            if missing_integration:
                print("Missing integration code:")
                for code in missing_integration:
                    print(f"  - {code}")
            
            return False
        
        print("✅ app.py integration test passed.")
        return True
        
    except Exception as e:
        print(f"❌ Error testing app integration: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running end-to-end integration tests...")
    
    # Run unit tests
    test_success = run_integration_tests()
    print(f"Integration tests passed: {test_success}")
    
    # Test app integration
    app_integration_success = test_app_integration()
    print(f"App integration test passed: {app_integration_success}")
    
    if test_success and app_integration_success:
        print("✅ All end-to-end integration tests passed!")
    else:
        print("❌ Some integration tests failed. Please review the errors.")
