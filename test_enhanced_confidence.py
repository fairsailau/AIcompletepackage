"""
Test script for the enhanced confidence framework.

This script validates the enhanced confidence framework implementation
for both document categorization and metadata extraction.
"""

import unittest
import sys
import os
import json
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modules to test
from modules.enhanced_confidence_framework import (
    calculate_enhanced_confidence,
    apply_category_specific_calibration,
    get_confidence_explanation,
    calculate_metadata_field_confidence,
    validate_metadata_cross_field,
    get_metadata_confidence_explanation
)

class TestEnhancedConfidenceFramework(unittest.TestCase):
    """Test cases for the enhanced confidence framework."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample document features
        self.document_features = {
            "file_size_kb": 500,
            "file_extension": ".pdf",
            "created_date": "2025-01-01",
            "modified_date": "2025-01-15"
        }
        
        # Sample valid categories
        self.valid_categories = [
            "Invoices", "Sales Contract", "Tax", 
            "Financial Report", "Employment Contract", "PII", "Other"
        ]
        
        # Sample category descriptions
        self.category_descriptions = {
            "Invoices": "Billing documents issued by a seller to a buyer, indicating quantities, prices for products or services.",
            "Sales Contract": "Contracts related to sales agreements and terms.",
            "Tax": "Documents related to government taxation (e.g., tax forms, filings, receipts).",
            "Financial Report": "Reports detailing the financial status or performance of an entity.",
            "Employment Contract": "Agreements outlining terms and conditions of employment.",
            "PII": "Documents containing Personally Identifiable Information that needs careful handling.",
            "Other": "Any document not fitting into the specific categories above."
        }
        
        # Sample document text
        self.document_text = """
        INVOICE
        
        Invoice Number: INV-12345
        Date: January 15, 2025
        Due Date: February 15, 2025
        
        Bill To:
        ACME Corporation
        123 Main Street
        Anytown, CA 12345
        
        Item Description | Quantity | Unit Price | Amount
        Widget A         | 10       | $25.00     | $250.00
        Widget B         | 5        | $30.00     | $150.00
        
        Subtotal: $400.00
        Tax (8%): $32.00
        Total: $432.00
        
        Payment Terms: Net 30
        """

    def test_calculate_enhanced_confidence(self):
        """Test the enhanced confidence calculation for document categorization."""
        # Test case for an invoice document
        result = calculate_enhanced_confidence(
            ai_reported_confidence=0.9,
            document_features=self.document_features,
            assigned_category="Invoices",
            reasoning="This document contains an invoice number, items with prices, and payment terms.",
            valid_categories=self.valid_categories,
            document_text=self.document_text,
            category_descriptions=self.category_descriptions
        )
        
        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertIn("semantic_alignment", result)
        self.assertIn("feature_correlation", result)
        self.assertIn("reasoning_coherence", result)
        self.assertIn("exclusion_confidence", result)
        self.assertIn("historical_performance", result)
        self.assertIn("overall", result)
        self.assertIn("ai_reported", result)
        
        # Verify the values are in the expected range
        for key, value in result.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        
        # For an invoice document with invoice-related text, semantic alignment should be high
        self.assertGreaterEqual(result["semantic_alignment"], 0.7)
        
        # Overall confidence should be reasonable for a good match
        self.assertGreaterEqual(result["overall"], 0.7)

    def test_apply_category_specific_calibration(self):
        """Test category-specific calibration."""
        # Test calibration for "Other" category (should reduce confidence)
        calibrated = apply_category_specific_calibration(
            category="Other",
            confidence=0.8,
            document_features=self.document_features
        )
        self.assertLess(calibrated, 0.8)
        
        # Test calibration for "Invoices" with PDF (should slightly boost confidence)
        calibrated = apply_category_specific_calibration(
            category="Invoices",
            confidence=0.8,
            document_features=self.document_features
        )
        self.assertGreaterEqual(calibrated, 0.8)

    def test_get_confidence_explanation(self):
        """Test generation of confidence explanations."""
        factors = {
            "semantic_alignment": 0.85,
            "feature_correlation": 0.75,
            "reasoning_coherence": 0.9,
            "exclusion_confidence": 0.7,
            "historical_performance": 0.8,
            "overall": 0.82,
            "ai_reported": 0.9
        }
        
        explanations = get_confidence_explanation(factors, "Invoices")
        
        # Verify the result structure
        self.assertIsInstance(explanations, dict)
        self.assertIn("semantic_alignment", explanations)
        self.assertIn("feature_correlation", explanations)
        self.assertIn("reasoning_coherence", explanations)
        self.assertIn("exclusion_confidence", explanations)
        self.assertIn("historical_performance", explanations)
        self.assertIn("overall", explanations)
        self.assertIn("ai_reported", explanations)
        
        # Verify explanations are non-empty strings
        for key, value in explanations.items():
            self.assertIsInstance(value, str)
            self.assertGreater(len(value), 10)

    def test_calculate_metadata_field_confidence(self):
        """Test confidence calculation for metadata fields."""
        # Test for invoice number field
        result = calculate_metadata_field_confidence(
            field_name="invoice_number",
            field_type="string",
            extracted_value="INV-12345",
            ai_confidence="High",
            document_text=self.document_text,
            field_definition={"name": "invoice_number", "type": "string"},
            validation_rules={"invoice_number": {"pattern": r"INV-\d+"}}
        )
        
        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertIn("numeric_confidence", result)
        self.assertIn("categorical_confidence", result)
        self.assertIn("factors", result)
        self.assertIn("original_ai_confidence", result)
        
        # Verify the confidence values
        self.assertGreaterEqual(result["numeric_confidence"], 0.0)
        self.assertLessEqual(result["numeric_confidence"], 1.0)
        self.assertIn(result["categorical_confidence"], ["High", "Medium", "Low"])
        
        # For a good invoice number match, confidence should be high
        self.assertGreaterEqual(result["numeric_confidence"], 0.7)
        
        # Test for a field not in the document
        result = calculate_metadata_field_confidence(
            field_name="customer_id",
            field_type="string",
            extracted_value="CUS-789",
            ai_confidence="Low",
            document_text=self.document_text
        )
        
        # Confidence should be lower for a field not in the document
        self.assertLess(result["numeric_confidence"], 0.7)
        self.assertEqual(result["categorical_confidence"], "Low")

    def test_validate_metadata_cross_field(self):
        """Test cross-field validation for metadata."""
        # Sample extracted fields
        extracted_fields = {
            "invoice_date": "2025-01-15",
            "due_date": "2025-02-15",
            "subtotal": "400.00",
            "tax_amount": "32.00",
            "total_amount": "432.00"
        }
        
        # Sample confidence scores
        confidence_scores = {
            "invoice_date": {
                "numeric_confidence": 0.8,
                "categorical_confidence": "High",
                "factors": {
                    "ai_reported": 0.9,
                    "format_validity": 0.9,
                    "content_plausibility": 0.8,
                    "extraction_consistency": 0.7,
                    "overall": 0.8
                }
            },
            "due_date": {
                "numeric_confidence": 0.7,
                "categorical_confidence": "Medium",
                "factors": {
                    "ai_reported": 0.6,
                    "format_validity": 0.9,
                    "content_plausibility": 0.7,
                    "extraction_consistency": 0.6,
                    "overall": 0.7
                }
            },
            "subtotal": {
                "numeric_confidence": 0.75,
                "categorical_confidence": "Medium",
                "factors": {
                    "ai_reported": 0.9,
                    "format_validity": 0.8,
                    "content_plausibility": 0.7,
                    "extraction_consistency": 0.6,
                    "overall": 0.75
                }
            },
            "tax_amount": {
                "numeric_confidence": 0.7,
                "categorical_confidence": "Medium",
                "factors": {
                    "ai_reported": 0.6,
                    "format_validity": 0.8,
                    "content_plausibility": 0.7,
                    "extraction_consistency": 0.6,
                    "overall": 0.7
                }
            },
            "total_amount": {
                "numeric_confidence": 0.8,
                "categorical_confidence": "High",
                "factors": {
                    "ai_reported": 0.9,
                    "format_validity": 0.9,
                    "content_plausibility": 0.8,
                    "extraction_consistency": 0.7,
                    "overall": 0.8
                }
            }
        }
        
        # Define relationships
        field_relationships = [
            {
                "fields": ["invoice_date", "due_date"],
                "relationship": "date_sequence",
                "description": "Due date should be after invoice date"
            },
            {
                "fields": ["subtotal", "tax_amount", "total_amount"],
                "relationship": "sum",
                "description": "Subtotal + tax should approximately equal total"
            }
        ]
        
        # Validate cross-field relationships
        updated_scores = validate_metadata_cross_field(
            extracted_fields=extracted_fields,
            confidence_scores=confidence_scores,
            field_relationships=field_relationships
        )
        
        # Verify the result structure
        self.assertIsInstance(updated_scores, dict)
        for field in extracted_fields:
            self.assertIn(field, updated_scores)
            self.assertIn("numeric_confidence", updated_scores[field])
            self.assertIn("categorical_confidence", updated_scores[field])
            self.assertIn("factors", updated_scores[field])
        
        # For valid relationships, confidence should be maintained or improved
        self.assertGreaterEqual(updated_scores["invoice_date"]["numeric_confidence"], 
                               confidence_scores["invoice_date"]["numeric_confidence"])
        self.assertGreaterEqual(updated_scores["due_date"]["numeric_confidence"], 
                               confidence_scores["due_date"]["numeric_confidence"])
        
        # Test with invalid relationships
        invalid_extracted_fields = {
            "invoice_date": "2025-02-15",  # Later than due date
            "due_date": "2025-01-15",      # Earlier than invoice date
            "subtotal": "400.00",
            "tax_amount": "32.00",
            "total_amount": "500.00"       # Doesn't match subtotal + tax
        }
        
        updated_scores = validate_metadata_cross_field(
            extracted_fields=invalid_extracted_fields,
            confidence_scores=confidence_scores,
            field_relationships=field_relationships
        )
        
        # For invalid relationships, confidence should be reduced
        self.assertLess(updated_scores["invoice_date"]["numeric_confidence"], 
                       confidence_scores["invoice_date"]["numeric_confidence"])
        self.assertLess(updated_scores["due_date"]["numeric_confidence"], 
                       confidence_scores["due_date"]["numeric_confidence"])
        self.assertLess(updated_scores["total_amount"]["numeric_confidence"], 
                       confidence_scores["total_amount"]["numeric_confidence"])

    def test_get_metadata_confidence_explanation(self):
        """Test generation of metadata confidence explanations."""
        confidence_result = {
            "numeric_confidence": 0.85,
            "categorical_confidence": "High",
            "factors": {
                "ai_reported": 0.9,
                "format_validity": 0.9,
                "content_plausibility": 0.8,
                "extraction_consistency": 0.7,
                "overall": 0.85
            }
        }
        
        explanations = get_metadata_confidence_explanation(confidence_result, "invoice_number")
        
        # Verify the result structure
        self.assertIsInstance(explanations, dict)
        self.assertIn("ai_reported", explanations)
        self.assertIn("format_validity", explanations)
        self.assertIn("content_plausibility", explanations)
        self.assertIn("extraction_consistency", explanations)
        self.assertIn("overall", explanations)
        
        # Verify explanations are non-empty strings
        for key, value in explanations.items():
            self.assertIsInstance(value, str)
            self.assertGreater(len(value), 10)


if __name__ == "__main__":
    unittest.main()
