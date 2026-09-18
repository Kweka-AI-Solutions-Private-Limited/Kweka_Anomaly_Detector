"""
leaf_disease.py — Schemas for Leaf Disease Analysis (Phase 1)
------------------------------------------------------------
Defines strict Pydantic schemas for image validation, crop identification,
normalized disease detection provider responses, diagnosis validation,
and final API payload structures.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from schemas.nacl_catalog import ProductRecommendationResult


class ImageValidationDetail(BaseModel):
    filename: str
    is_valid: bool
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: int = 0
    blur_score: float = 0.0
    brightness: float = 0.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class OverallImageValidationResult(BaseModel):
    total_submitted: int
    valid_count: int
    invalid_count: int
    is_any_valid: bool
    details: List[ImageValidationDetail] = Field(default_factory=list)


class CropIdentificationResult(BaseModel):
    crop_name: str
    confidence: Optional[float] = None
    source: str = Field(description="'user_selected', 'automatic', or 'uncertain'")
    status: str = Field(description="'CONFIRMED', 'USER_SPECIFIED', or 'UNCERTAIN'")
    evidence_note: str


class DiseaseCandidate(BaseModel):
    disease_name: str
    confidence: Optional[float] = None


class NormalizedDiseaseResult(BaseModel):
    disease_name: str
    confidence: Optional[float] = None
    provider_name: str
    is_mock: bool = False
    supported: bool = True
    candidates: List[DiseaseCandidate] = Field(default_factory=list)
    provider_status: str = Field(description="'OK', 'UNSUPPORTED_CROP', 'LOW_CONFIDENCE', 'PROVIDER_ERROR', 'MOCK'")
    error_message: Optional[str] = None


class DiagnosisValidationResult(BaseModel):
    evidence_level: str = Field(description="'HIGH', 'MEDIUM', or 'LOW'")
    image_quality_status: str
    crop_compatibility_status: str
    consistency_status: str
    is_uncertain: bool
    validation_notes: List[str] = Field(default_factory=list)


class LeafDiseaseAnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: str
    status: str = Field(description="'SUCCESS', 'UNCERTAIN', or 'FAILED'")
    validation: OverallImageValidationResult
    crop: CropIdentificationResult
    disease: Optional[NormalizedDiseaseResult] = None
    diagnosis: Optional[DiagnosisValidationResult] = None
    nacl_recommendations: Optional[ProductRecommendationResult] = None
    retry_guidance: Optional[str] = None
    phase_notice: str = "Phase 3: Integrated Leaf Disease Detection & RAG NACL Agrochemical Product Recommendations."
