"""
InspectAI MongoDB Collection Schemas & Data Models
---------------------------------------------------
Defines Pydantic models and document conversion helpers for the 6 core collections:
  1. models
  2. model_versions
  3. reference_images
  4. inspections
  5. inspection_results
  6. feedback
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, values=None, **kwargs):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


# ------------------------------------------------------------
# 0. MODEL GROUPS
# ------------------------------------------------------------
class ModelGroupSchema(BaseModel):
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}


class ModelGroupCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ModelGroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


# ------------------------------------------------------------
# 1. MODELS
# ------------------------------------------------------------
class ModelSchema(BaseModel):
    name: str
    description: Optional[str] = None
    status: str = "draft"  # draft, building, inactive, active, error
    error_reason: Optional[str] = None
    model_type: str = "patchcore"
    domain: Optional[str] = None
    group_id: Optional[Any] = None  # ObjectId
    reference_image_count: int = 0
    active_version_id: Optional[Any] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}


# ------------------------------------------------------------
# 2. MODEL VERSIONS
# ------------------------------------------------------------
class AlgorithmConfig(BaseModel):
    name: str = "patchcore"
    backbone: str = "wide_resnet50_2"
    layers: List[str] = ["layer2"]
    coreset_sampling_ratio: float = 0.05
    num_neighbors: int = 9
    image_size: List[int] = [256, 256]


class CalibrationConfig(BaseModel):
    method: str = "95th_percentile"
    threshold: float = 27.0
    auto_calibrated_threshold: Optional[float] = None
    is_custom: bool = False


class TrainingConfig(BaseModel):
    reference_count: int = 0
    reference_ids: List[Any] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    build_time_ms: float = 0.0


class VersionArtifacts(BaseModel):
    checkpoint_uri: Optional[str] = None
    memory_bank_uri: Optional[str] = None
    metadata_uri: Optional[str] = None


class ModelVersionSchema(BaseModel):
    model_id: Any  # ObjectId
    version_number: int
    status: str = "building"  # building, ready, active, archived, failed
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    artifacts: VersionArtifacts = Field(default_factory=VersionArtifacts)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}


# ------------------------------------------------------------
# 3. REFERENCE IMAGES
# ------------------------------------------------------------
class ImageStorage(BaseModel):
    uri: str


class ReferenceImageSchema(BaseModel):
    model_id: Any  # ObjectId
    version_id: Any  # ObjectId
    type: str = "good"
    storage: ImageStorage
    filename: str
    relative_path: str
    width: int = 256
    height: int = 256
    file_size: int = 0
    checksum: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}


# ------------------------------------------------------------
# 4. INSPECTIONS & RUNS
# ------------------------------------------------------------
class RunSummary(BaseModel):
    total: int = 0
    pass_count: int = 0
    reject_count: int = 0
    errors: int = 0


class InspectionRunSchema(BaseModel):
    model_id: Any  # ObjectId
    model_version_id: Any  # ObjectId
    run_number: int  # Chronological per model: 1, 2, 3...
    status: str = "running"  # running, completed, partial, failed
    total_images: int = 0
    completed_images: int = 0
    pass_count: int = 0
    reject_count: int = 0
    error_count: int = 0
    summary: RunSummary = Field(default_factory=RunSummary)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    class Config:
        json_encoders = {ObjectId: str}


# ------------------------------------------------------------
# SAFETY & MULTI-INSTANCE LIMIT CONSTANTS
# ------------------------------------------------------------
MAX_IMAGE_DIMENSION = 4096
MAX_INSTANCES = 20
MIN_INSTANCE_AREA = 500


class InspectionInput(BaseModel):
    storage_uri: str
    filename: str
    width: int = 256
    height: int = 256


class InspectionSchema(BaseModel):
    model_id: Any  # ObjectId
    model_version_id: Any  # ObjectId
    run_id: Optional[Any] = None  # ObjectId (Parent InspectionRun if part of batch run)
    inspection_mode: str = "single"  # single, multi_instance
    status: str = "queued"  # queued, processing, completed, failed, review
    input: InspectionInput
    processing_time_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    class Config:
        json_encoders = {ObjectId: str}


# ------------------------------------------------------------
# 5. INSPECTION RESULTS
# ------------------------------------------------------------
class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class PredictionOutput(BaseModel):
    status: str  # normal, anomalous, review, PASS, REJECT
    anomaly_score: float
    threshold: float
    severity: Optional[str] = None


class OverallPredictionOutput(BaseModel):
    status: str  # PASS, REJECT, REVIEW
    total_instances: int = 0
    pass_count: int = 0
    reject_count: int = 0
    error_count: int = 0
    max_anomaly_score: Optional[float] = None
    threshold: Optional[float] = None
    reason: Optional[str] = None  # NO_OBJECTS_DETECTED, INSTANCE_DETECTION_FAILED, PATCHCORE_INSTANCE_FAILED
    message: Optional[str] = None


class LocalizationOutput(BaseModel):
    bbox: Optional[BoundingBox] = None
    heatmap_uri: Optional[str] = None
    mask_uri: Optional[str] = None


class ExplanationOutput(BaseModel):
    defect_type: Optional[str] = None
    location: Optional[str] = None
    explanation: Optional[str] = None


class VLMAnalysisSchema(BaseModel):
    status: str = "completed"  # completed, unavailable, failed, skipped, generating
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    defect_type: Optional[str] = "Unknown / Insufficient visual evidence"
    location: Optional[str] = None
    severity: Optional[str] = "Low"  # Low, Medium, High, Critical
    prominence: Optional[str] = None
    explanation: Optional[str] = None
    visual_evidence: Optional[str] = None
    confidence: Optional[float] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class InstanceResultSchema(BaseModel):
    instance_id: int
    bbox: BoundingBox  # Original image coordinate space
    padded_bbox: Optional[BoundingBox] = None
    crop_storage_uri: str
    detection_confidence: float = 1.0
    status: str = "completed"  # completed, failed
    error: Optional[str] = None
    prediction: Optional[PredictionOutput] = None
    localization: Optional[LocalizationOutput] = None
    vlm_analysis: Optional[VLMAnalysisSchema] = None


class InspectionResultSchema(BaseModel):
    inspection_id: Any  # ObjectId (UNIQUE)
    inspection_mode: str = "single"  # single, multi_instance
    prediction: Optional[PredictionOutput] = None
    overall_prediction: Optional[OverallPredictionOutput] = None
    localization: Optional[LocalizationOutput] = None
    instances: List[InstanceResultSchema] = Field(default_factory=list)
    composite_heatmap_uri: Optional[str] = None
    explanation: ExplanationOutput = Field(default_factory=ExplanationOutput)
    vlm_analysis: Optional[VLMAnalysisSchema] = None
    processing_stats: Optional[Dict[str, float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}


# ------------------------------------------------------------
# 6. FEEDBACK
# ------------------------------------------------------------
SUPPORTED_DETECTION_FEEDBACK = ["correct", "false_positive", "false_negative"]
SUPPORTED_VLM_FEEDBACK_CATEGORIES = ["correct", "wrong_defect_type", "wrong_location", "wrong_severity"]
SUPPORTED_SEVERITIES = ["Low", "Medium", "High", "Critical"]


class FeedbackSchema(BaseModel):
    inspection_id: Any  # ObjectId
    inspection_result_id: Optional[Any] = None  # ObjectId
    model_id: Any  # ObjectId
    model_version_id: Any  # ObjectId
    run_id: Optional[Any] = None  # ObjectId

    # Original Prediction Snapshots (Preserves historical record without modification)
    original_prediction: Optional[Dict[str, Any]] = None
    original_vlm_analysis: Optional[Dict[str, Any]] = None

    # Detection Feedback (PatchCore)
    # REJECT: "correct", "false_positive"
    # PASS: "correct", "false_negative"
    detection_feedback: str = "correct"

    # VLM Interpretation Feedback (Gemini - multi-select allowed)
    # "correct", "wrong_defect_type", "wrong_location", "wrong_severity"
    vlm_feedback_categories: List[str] = Field(default_factory=list)

    # Corrected Values
    corrected_defect_type: Optional[str] = None
    corrected_location: Optional[str] = None
    corrected_severity: Optional[str] = None

    # Additional Comment
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}


# ------------------------------------------------------------
# 7. NOTIFICATIONS
# ------------------------------------------------------------
class NotificationSchema(BaseModel):
    type: str  # MODEL_BUILD_COMPLETED, MODEL_BUILD_FAILED, INSPECTION_RUN_COMPLETED, INSPECTION_RUN_PARTIAL, INSPECTION_RUN_FAILED
    title: str
    message: str
    severity: str = "info"  # success, warning, error, info
    related_model_id: Optional[Any] = None  # ObjectId
    related_version_id: Optional[Any] = None  # ObjectId
    related_run_id: Optional[Any] = None  # ObjectId
    target_route: str = "/"
    read: bool = False
    idempotency_key: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}


