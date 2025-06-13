from enum import Enum
from dataclasses import dataclass, field # Import field for potential default_factory
from typing import Dict, Any, List, Optional

class ProcessingStatus(Enum):
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
    auto_approve_threshold: float = 0.85
    human_review_threshold: float = 0.6
    auto_reject_threshold: float = 0.3
    max_files_per_batch: int = 50
    max_processing_time_minutes: int = 30
    escalate_on_category_change: bool = True
    escalate_on_low_metadata_confidence: bool = True
    escalate_on_field_inconsistency: bool = True
    human_review_timeout_hours: int = 24

@dataclass
class DocumentProcessingResult:
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
