"""
diagnosis_validator.py — Evidence Evaluation & Diagnosis Validation Layer (Phase 1)
---------------------------------------------------------------------------------
Evaluates detection evidence conservatively based on:
  - Disease provider confidence
  - Image quality (blur, resolution, exposure)
  - Crop compatibility
  - Multi-image consistency

Classifies diagnosis evidence into HIGH, MEDIUM, or LOW without inventing
synthetic mathematical probability equations.
"""

from typing import List, Optional
from schemas.leaf_disease import (
    OverallImageValidationResult,
    CropIdentificationResult,
    NormalizedDiseaseResult,
    DiagnosisValidationResult,
)

PATHOGEN_HOST_MAP = {
    "podosphaera xanthii": ["cucurbits", "cucumber", "melon", "squash", "pumpkin", "gourd"],
    "golovinomyces cichoracearum": ["cucurbits", "cucumber", "melon", "squash", "pumpkin", "gourd"],
    "erysiphe necator": ["grape"],
    "uncinula necator": ["grape"],
    "gymnosporangium sabinae": ["pear"],
    "gymnosporangium juniperi-virginianae": ["apple"],
}


def _check_host_compatibility(crop_name: str, disease_name: str) -> Optional[bool]:
    """
    Returns True if compatible, False if host-pathogen mismatch, or None if no obligate host constraint exists.
    """
    if not crop_name or not disease_name:
        return None

    disease_lower = disease_name.lower()
    crop_lower = crop_name.lower()

    for pathogen_key, valid_hosts in PATHOGEN_HOST_MAP.items():
        if pathogen_key in disease_lower:
            # Check if current crop is among valid hosts
            is_valid_host = any(vh in crop_lower for vh in valid_hosts)
            return is_valid_host

    return None


def validate_diagnosis_evidence(
    validation_result: OverallImageValidationResult,
    crop_info: CropIdentificationResult,
    disease_result: NormalizedDiseaseResult
) -> DiagnosisValidationResult:
    """
    Evaluates evidence factors conservatively and returns qualitative evidence classification.
    Strictly enforces host-pathogen compatibility constraints.
    """
    notes: List[str] = []
    
    # 1. Image Quality Evaluation
    valid_details = [d for d in validation_result.details if d.is_valid]
    has_blurry = any(d.blur_score < 25.0 for d in valid_details)
    has_exposure_warnings = any("dark" in w.lower() or "overexposed" in w.lower() for d in valid_details for w in d.warnings)

    if not valid_details:
        image_quality_status = "POOR"
        notes.append("No valid images available for assessment.")
    elif has_blurry:
        image_quality_status = "SUBOPTIMAL"
        notes.append("Some images exhibit blur or softness.")
    elif has_exposure_warnings:
        image_quality_status = "ACCEPTABLE"
        notes.append("Image lighting/exposure is acceptable but suboptimal.")
    else:
        image_quality_status = "GOOD"
        notes.append("Image quality and clarity are good.")

    # 2. Host-Pathogen Compatibility Check
    host_compat = _check_host_compatibility(crop_info.crop_name, disease_result.disease_name)

    if host_compat is False:
        crop_compatibility_status = "MISMATCHED_HOST"
        notes.append(
            f"Host-pathogen mismatch detected: Detected pathogen '{disease_result.disease_name}' "
            f"is incompatible with identified crop '{crop_info.crop_name}'."
        )
    elif crop_info.status in ("CONFIRMED", "USER_SPECIFIED"):
        crop_compatibility_status = "COMPATIBLE"
        notes.append(f"Crop '{crop_info.crop_name}' verified ({crop_info.status}).")
    else:
        crop_compatibility_status = "UNCERTAIN"
        notes.append(f"Crop identification is uncertain.")

    # 3. Consistency Evaluation
    if validation_result.valid_count > 1:
        consistency_status = "CONSISTENT"
        notes.append(f"Analysis consistent across {validation_result.valid_count} submitted images.")
    else:
        consistency_status = "SINGLE_IMAGE"
        notes.append("Single image analysis submitted.")

    # 4. Evidence Level Categorization (HIGH / MEDIUM / LOW)
    conf = disease_result.confidence
    provider_ok = disease_result.provider_status in ("OK", "MOCK") and disease_result.supported

    if crop_compatibility_status == "MISMATCHED_HOST":
        evidence_level = "LOW"
        is_uncertain = True
        notes.append("Evidence level demoted to LOW due to host-pathogen mismatch constraint.")
    elif not provider_ok or conf is None:
        evidence_level = "LOW"
        is_uncertain = True
        notes.append("Disease detection provider returned unconfirmed or error status.")
    elif conf >= 0.85 and image_quality_status in ("GOOD", "ACCEPTABLE") and crop_compatibility_status == "COMPATIBLE":
        evidence_level = "HIGH"
        is_uncertain = False
        notes.append("High evidence: strong provider confidence, valid crop, and clear image quality.")
    elif conf >= 0.70 and image_quality_status != "POOR" and crop_compatibility_status == "COMPATIBLE":
        evidence_level = "MEDIUM"
        is_uncertain = False
        notes.append("Medium evidence: moderate provider confidence or minor image quality warnings.")
    else:
        evidence_level = "LOW"
        is_uncertain = True
        notes.append("Low evidence: provider confidence is low or image quality is suboptimal.")

    return DiagnosisValidationResult(
        evidence_level=evidence_level,
        image_quality_status=image_quality_status,
        crop_compatibility_status=crop_compatibility_status,
        consistency_status=consistency_status,
        is_uncertain=is_uncertain,
        validation_notes=notes
    )
