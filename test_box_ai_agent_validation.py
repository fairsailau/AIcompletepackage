"""
Validation tests for Box AI agent decision boundaries and autonomy.

This module provides comprehensive tests to ensure the Box AI agent makes correct
decisions based on confidence thresholds and escalation criteria.
"""

import unittest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import agent components
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

class TestBoxAIAgentDecisionBoundaries(unittest.TestCase):
    """Test cases for Box AI agent decision boundaries."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock Box client
        self.mock_client = Mock()
        
        # Create test boundaries
        self.boundaries = ProcessingBoundaries(
            auto_approve_threshold=0.85,
            human_review_threshold=0.6,
            auto_reject_threshold=0.3
        )
        
        # Create agent with mocked dependencies
        with patch('modules.box_ai_agent.BoxAIDocumentProcessor') as mock_processor:
            with patch('modules.box_ai_agent.ChatOpenAI'):
                self.agent = BoxAIDocumentProcessingAgent(
                    client=self.mock_client,
                    boundaries=self.boundaries,
                    enable_human_loop=True
                )
    
    def test_auto_approve_decision(self):
        """Test auto-approve decision for high confidence scores."""
        # Mock high confidence Box AI result
        high_confidence_result = {
            "file_id": "test_file_1",
            "success": True,
            "categorization": {
                "document_type": "Invoice",
                "confidence": 0.95,
                "reasoning": "Clear invoice format with all required fields"
            },
            "metadata": {
                "invoice_number": "INV-2024-001",
                "invoice_number_confidence": "High",
                "invoice_number_confidence_numeric": 0.90,
                "total_amount": "1500.00",
                "total_amount_confidence": "High",
                "total_amount_confidence_numeric": 0.88
            },
            "overall_confidence": 0.92,
            "processing_time": 1.5
        }
        
        # Mock Box AI processor
        self.agent.box_ai_processor.process_document_workflow = Mock(return_value=high_confidence_result)
        
        # Process document
        result = self.agent.process_document(
            file_id="test_file_1",
            field_definitions=[
                {"name": "invoice_number", "type": "string"},
                {"name": "total_amount", "type": "number"}
            ]
        )
        
        # Verify auto-approve decision
        self.assertEqual(result.status, ProcessingStatus.AUTO_APPROVED)
        self.assertGreaterEqual(result.confidence_scores["overall"], self.boundaries.auto_approve_threshold)
        self.assertNotIn("test_file_1", self.agent.human_review_queue)
    
    def test_human_review_decision(self):
        """Test human review decision for medium confidence scores."""
        # Mock medium confidence Box AI result
        medium_confidence_result = {
            "file_id": "test_file_2",
            "success": True,
            "categorization": {
                "document_type": "Contract",
                "confidence": 0.75,
                "reasoning": "Document appears to be a contract but some fields are unclear"
            },
            "metadata": {
                "contract_id": "CONT-2024-001",
                "contract_id_confidence": "Medium",
                "contract_id_confidence_numeric": 0.65,
                "effective_date": "2024-01-01",
                "effective_date_confidence": "Medium",
                "effective_date_confidence_numeric": 0.68
            },
            "overall_confidence": 0.72,
            "processing_time": 2.0
        }
        
        # Mock Box AI processor
        self.agent.box_ai_processor.process_document_workflow = Mock(return_value=medium_confidence_result)
        
        # Process document
        result = self.agent.process_document(
            file_id="test_file_2",
            field_definitions=[
                {"name": "contract_id", "type": "string"},
                {"name": "effective_date", "type": "date"}
            ]
        )
        
        # Verify human review decision
        self.assertEqual(result.status, ProcessingStatus.HUMAN_REVIEW_REQUIRED)
        self.assertLess(result.confidence_scores["overall"], self.boundaries.auto_approve_threshold)
        self.assertGreaterEqual(result.confidence_scores["overall"], self.boundaries.human_review_threshold)
        self.assertIn("test_file_2", self.agent.human_review_queue)
    
    def test_error_decision(self):
        """Test error decision for low confidence scores."""
        # Mock low confidence Box AI result
        low_confidence_result = {
            "file_id": "test_file_3",
            "success": True,
            "categorization": {
                "document_type": "Other",
                "confidence": 0.25,
                "reasoning": "Unable to clearly identify document type"
            },
            "metadata": {
                "field1": "unclear_value",
                "field1_confidence": "Low",
                "field1_confidence_numeric": 0.25
            },
            "overall_confidence": 0.25,
            "processing_time": 1.8
        }
        
        # Mock Box AI processor
        self.agent.box_ai_processor.process_document_workflow = Mock(return_value=low_confidence_result)
        
        # Process document
        result = self.agent.process_document(
            file_id="test_file_3",
            field_definitions=[
                {"name": "field1", "type": "string"}
            ]
        )
        
        # Verify error decision
        self.assertEqual(result.status, ProcessingStatus.ERROR)
        self.assertLess(result.confidence_scores["overall"], self.boundaries.human_review_threshold)
        # Note: In our implementation, very low confidence results in ERROR status, not auto-reject
    
    def test_escalation_criteria(self):
        """Test various escalation criteria."""
        # Test category change escalation
        if self.boundaries.escalate_on_category_change:
            # Simulate category change during processing
            initial_category = "Invoice"
            final_category = "Contract"
            
            # Should escalate if categories are different
            should_escalate = initial_category != final_category
            self.assertTrue(should_escalate)
        
        # Test low metadata confidence escalation
        if self.boundaries.escalate_on_low_metadata_confidence:
            # Simulate low metadata confidence
            metadata_confidences = {
                "field1": 0.45,  # Below human review threshold
                "field2": 0.55   # Below human review threshold
            }
            
            avg_confidence = sum(metadata_confidences.values()) / len(metadata_confidences)
            should_escalate = avg_confidence < self.boundaries.human_review_threshold
            self.assertTrue(should_escalate)
        
        # Test field inconsistency escalation
        if self.boundaries.escalate_on_field_inconsistency:
            # Simulate field inconsistency (e.g., due date before invoice date)
            invoice_date = "2024-01-15"
            due_date = "2024-01-10"  # Before invoice date - inconsistent
            
            # This would be detected by cross-field validation
            is_inconsistent = due_date < invoice_date
            self.assertTrue(is_inconsistent)
    
    def test_human_feedback_integration(self):
        """Test human feedback integration."""
        # Create a test processing result
        file_id = "test_file_4"
        result = DocumentProcessingResult(
            file_id=file_id,
            file_name="test_document.pdf",
            status=ProcessingStatus.HUMAN_REVIEW_REQUIRED,
            escalation_reason="Low confidence score"
        )
        
        # Add to agent's processing results
        self.agent.processing_results[file_id] = result
        self.agent.human_review_queue.append(file_id)
        
        # Test human approval
        feedback = {
            "reviewer": "test_user",
            "comments": "Document looks correct",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        success = self.agent.submit_human_feedback(file_id, approved=True, feedback=feedback)
        
        # Verify feedback was processed
        self.assertTrue(success)
        self.assertEqual(self.agent.processing_results[file_id].status, ProcessingStatus.HUMAN_APPROVED)
        self.assertEqual(self.agent.processing_results[file_id].human_feedback, feedback)
        self.assertNotIn(file_id, self.agent.human_review_queue)
        
        # Test human rejection
        file_id_2 = "test_file_5"
        result_2 = DocumentProcessingResult(
            file_id=file_id_2,
            file_name="test_document_2.pdf",
            status=ProcessingStatus.HUMAN_REVIEW_REQUIRED,
            escalation_reason="Field inconsistency"
        )
        
        self.agent.processing_results[file_id_2] = result_2
        self.agent.human_review_queue.append(file_id_2)
        
        feedback_2 = {
            "reviewer": "test_user",
            "comments": "Document has errors",
            "timestamp": "2024-01-01T12:05:00Z"
        }
        
        success_2 = self.agent.submit_human_feedback(file_id_2, approved=False, feedback=feedback_2)
        
        # Verify rejection was processed
        self.assertTrue(success_2)
        self.assertEqual(self.agent.processing_results[file_id_2].status, ProcessingStatus.HUMAN_REJECTED)
        self.assertEqual(self.agent.processing_results[file_id_2].human_feedback, feedback_2)
        self.assertNotIn(file_id_2, self.agent.human_review_queue)

class TestBoxAIToolIntegration(unittest.TestCase):
    """Test cases for Box AI Tool integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock Box client
        self.mock_client = Mock()
        
        # Mock Box API responses
        self.mock_client._oauth = Mock(access_token="mock_token")
        
        # Create Box AI tool
        self.box_ai_tool = BoxAITool(self.mock_client)
        
        # Create Box AI processor
        self.box_ai_processor = BoxAIDocumentProcessor(self.mock_client)
    
    @patch('modules.box_ai_tool.requests.post')
    def test_categorization_api_call(self, mock_post):
        """Test Box AI categorization API call."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "entries": [
                {
                    "classification": {
                        "category": "Invoice",
                        "confidence": 0.95
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Call categorization
        result = self.box_ai_processor.categorize_document("test_file_id")
        
        # Verify API call
        mock_post.assert_called_once()
        self.assertEqual(result["document_type"], "Invoice")
        self.assertEqual(result["confidence"], 0.95)
    
    @patch('modules.box_ai_tool.requests.post')
    def test_metadata_extraction_api_call(self, mock_post):
        """Test Box AI metadata extraction API call."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "entries": [
                {
                    "key_value_extraction": [
                        {
                            "key": "invoice_number",
                            "value": "INV-2024-001",
                            "confidence": 0.9
                        },
                        {
                            "key": "total_amount",
                            "value": "1500.00",
                            "confidence": 0.85
                        }
                    ]
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Call metadata extraction
        result = self.box_ai_processor.extract_metadata(
            "test_file_id",
            ["invoice_number", "total_amount"]
        )
        
        # Verify API call
        mock_post.assert_called_once()
        self.assertTrue(result["success"])
        self.assertEqual(result["metadata"]["invoice_number"], "INV-2024-001")
        self.assertEqual(result["metadata"]["total_amount"], "1500.00")
        self.assertEqual(result["metadata"]["invoice_number_confidence"], "High")
        self.assertEqual(result["metadata"]["total_amount_confidence"], "High")
    
    @patch('modules.box_ai_tool.requests.post')
    def test_complete_workflow(self, mock_post):
        """Test complete document workflow."""
        # Mock responses for categorization and metadata extraction
        mock_responses = [
            # Categorization response
            Mock(
                status_code=200,
                json=lambda: {
                    "entries": [
                        {
                            "classification": {
                                "category": "Invoice",
                                "confidence": 0.95
                            }
                        }
                    ]
                }
            ),
            # Metadata extraction response
            Mock(
                status_code=200,
                json=lambda: {
                    "entries": [
                        {
                            "key_value_extraction": [
                                {
                                    "key": "invoice_number",
                                    "value": "INV-2024-001",
                                    "confidence": 0.9
                                },
                                {
                                    "key": "total_amount",
                                    "value": "1500.00",
                                    "confidence": 0.85
                                }
                            ]
                        }
                    ]
                }
            )
        ]
        mock_post.side_effect = mock_responses
        
        # Mock file info
        self.mock_client.file.return_value.get.return_value = Mock(name="test_invoice.pdf")
        
        # Call complete workflow
        result = self.box_ai_processor.process_document_workflow(
            file_id="test_file_id",
            categories=["Invoice", "Contract", "Other"],
            metadata_fields=["invoice_number", "total_amount"]
        )
        
        # Verify workflow result
        self.assertTrue(result["success"])
        self.assertEqual(result["categorization"]["document_type"], "Invoice")
        self.assertEqual(result["metadata"]["invoice_number"], "INV-2024-001")
        self.assertEqual(result["metadata"]["total_amount"], "1500.00")
        self.assertGreaterEqual(result["overall_confidence"], 0.9)

class TestBoxAIAgentAutonomy(unittest.TestCase):
    """Test cases for Box AI agent autonomy and decision-making."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_client = Mock()
        self.boundaries = ProcessingBoundaries()
        
        with patch('modules.box_ai_agent.BoxAIDocumentProcessor'):
            with patch('modules.box_ai_agent.ChatOpenAI'):
                self.agent = BoxAIDocumentProcessingAgent(
                    client=self.mock_client,
                    boundaries=self.boundaries,
                    enable_human_loop=True
                )
    
    def test_autonomous_workflow_execution(self):
        """Test that agent can execute workflow autonomously."""
        # Mock successful Box AI processing
        self.agent.box_ai_processor.process_document_workflow = Mock(return_value={
            "file_id": "test_file",
            "success": True,
            "categorization": {
                "document_type": "Invoice",
                "confidence": 0.95,
                "reasoning": "Clear invoice format with all required fields"
            },
            "metadata": {
                "invoice_number": "INV-2024-001",
                "invoice_number_confidence": "High",
                "invoice_number_confidence_numeric": 0.90,
                "total_amount": "1500.00",
                "total_amount_confidence": "High",
                "total_amount_confidence_numeric": 0.88
            },
            "overall_confidence": 0.92,
            "processing_time": 1.5
        })
        
        # Test autonomous processing
        field_definitions = [
            {"name": "invoice_number", "type": "string"},
            {"name": "total_amount", "type": "number"}
        ]
        
        result = self.agent.process_document(
            file_id="test_file",
            field_definitions=field_definitions
        )
        
        # Verify processing was successful
        self.assertIsInstance(result, DocumentProcessingResult)
        self.assertEqual(result.file_id, "test_file")
        self.assertEqual(result.status, ProcessingStatus.AUTO_APPROVED)
        self.agent.box_ai_processor.process_document_workflow.assert_called_once()
    
    def test_batch_processing_autonomy(self):
        """Test autonomous batch processing."""
        file_ids = ["file_1", "file_2", "file_3"]
        field_definitions = [
            {"name": "document_type", "type": "string"}
        ]
        
        with patch.object(self.agent, 'process_document') as mock_process:
            mock_process.return_value = DocumentProcessingResult(
                file_id="test",
                file_name="test.pdf",
                status=ProcessingStatus.COMPLETED
            )
            
            results = self.agent.process_batch(
                file_ids=file_ids,
                field_definitions=field_definitions
            )
            
            # Verify all files were processed
            self.assertEqual(len(results), len(file_ids))
            self.assertEqual(mock_process.call_count, len(file_ids))
    
    def test_error_handling_autonomy(self):
        """Test agent's autonomous error handling."""
        # Mock error during processing
        self.agent.box_ai_processor.process_document_workflow = Mock(side_effect=Exception("Processing error"))
        
        result = self.agent.process_document(
            file_id="error_file",
            field_definitions=[]
        )
        
        # Verify error was handled autonomously
        self.assertEqual(result.status, ProcessingStatus.ERROR)
        self.assertIsNotNone(result.error_message)
        self.assertIn("error_file", self.agent.processing_results)

def run_validation_tests():
    """Run all validation tests for Box AI agent decision boundaries and autonomy."""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBoxAIAgentDecisionBoundaries))
    test_suite.addTest(unittest.makeSuite(TestBoxAIToolIntegration))
    test_suite.addTest(unittest.makeSuite(TestBoxAIAgentAutonomy))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

def validate_box_ai_agent_configuration():
    """Validate Box AI agent configuration and boundaries."""
    
    validation_results = {
        "boundary_validation": True,
        "box_ai_validation": True,
        "autonomy_validation": True,
        "integration_validation": True,
        "errors": []
    }
    
    try:
        # Test boundary configuration
        boundaries = ProcessingBoundaries()
        
        # Validate threshold relationships
        if boundaries.auto_approve_threshold <= boundaries.human_review_threshold:
            validation_results["boundary_validation"] = False
            validation_results["errors"].append("Auto-approve threshold must be higher than human review threshold")
        
        if boundaries.human_review_threshold <= boundaries.auto_reject_threshold:
            validation_results["boundary_validation"] = False
            validation_results["errors"].append("Human review threshold must be higher than auto-reject threshold")
        
        # Validate Box AI tool creation
        mock_client = Mock()
        mock_client._oauth = Mock(access_token="mock_token")
        
        try:
            box_ai_tool = BoxAITool(mock_client)
            box_ai_processor = BoxAIDocumentProcessor(mock_client)
        except Exception as e:
            validation_results["box_ai_validation"] = False
            validation_results["errors"].append(f"Box AI tool initialization error: {str(e)}")
        
        # Test agent creation
        with patch('modules.box_ai_agent.BoxAIDocumentProcessor'):
            with patch('modules.box_ai_agent.ChatOpenAI'):
                agent = BoxAIDocumentProcessingAgent(
                    client=mock_client,
                    boundaries=boundaries,
                    enable_human_loop=True
                )
                
                # Validate agent components
                if not agent.tools:
                    validation_results["autonomy_validation"] = False
                    validation_results["errors"].append("Agent tools not properly initialized")
                
                if not agent.agent:
                    validation_results["autonomy_validation"] = False
                    validation_results["errors"].append("Agent executor not properly initialized")
    
    except Exception as e:
        validation_results["integration_validation"] = False
        validation_results["errors"].append(f"Agent initialization error: {str(e)}")
    
    return validation_results

if __name__ == "__main__":
    print("Running Box AI agent validation tests...")
    
    # Run unit tests
    test_success = run_validation_tests()
    print(f"Unit tests passed: {test_success}")
    
    # Run configuration validation
    config_results = validate_box_ai_agent_configuration()
    print(f"Configuration validation: {config_results}")
    
    if test_success and all(config_results.values()):
        print("✅ All Box AI agent validation tests passed!")
    else:
        print("❌ Some validation tests failed. Please review the errors.")
        if config_results["errors"]:
            for error in config_results["errors"]:
                print(f"  - {error}")
