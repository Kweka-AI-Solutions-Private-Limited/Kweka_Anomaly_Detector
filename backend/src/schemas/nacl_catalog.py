"""
nacl_catalog.py — Pydantic Schemas for NACL Agrochemical Catalog (Phase 2 & 3)
-------------------------------------------------------------------------------
Defines product catalog models and recommendation response structures for NACL
fungicides, insecticides, herbicides, and plant growth regulators (PGRs).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CropDosageRule(BaseModel):
    """Specific crop application rule, dosage, and target pest/disease."""
    crop: str = Field(..., description="Target crop or plant species, e.g. Rice, Chilli, Pear, Apple, Cucurbits")
    target_pest_or_disease: Optional[str] = Field(None, description="Target insect pest, fungal disease, or weed")
    dosage: Optional[str] = Field(None, description="Recommended application dosage, e.g. 120 gm/acre or 600 ml/acre")
    dosage_verified: bool = Field(False, description="True if dosage is verified from authoritative source for exact crop+target")
    source_url: Optional[str] = Field(None, description="Source URL where crop dosage was verified")
    source_type: Optional[str] = Field(None, description="OFFICIAL_NACL_LABEL, OFFICIAL_NACL_PAGE, REGULATORY")


class NACLProduct(BaseModel):
    """Complete structured model for an NACL agrochemical product."""
    product_id: str = Field(..., description="Unique slug or ID derived from product URL")
    product_name: str = Field(..., description="Brand name of the product, e.g. Task SC, Nagarjuna Suraksha")
    category: str = Field(..., description="Agrochemical category: Insecticides, Fungicides, Herbicides, PGR")
    product_url: str = Field(..., description="Direct URL to official NACL product web page")
    active_ingredient: Optional[str] = Field(None, description="Active chemical ingredient & concentration, e.g. Fipronil 5% SC")
    chemical_class: Optional[str] = Field(None, description="Chemical family/group, e.g. Phenylpyrazole, Triazole")
    frac_group: Optional[str] = Field(None, description="FRAC mode of action code/group, e.g. Group 1, Group 11 + M5")
    mode_of_action: Optional[str] = Field(None, description="Detailed biological mode of action")
    pack_sizes: List[str] = Field(default_factory=list, description="Available commercial packaging sizes")
    key_benefits: List[str] = Field(default_factory=list, description="Key product feature bullet points")
    crop_applications: List[CropDosageRule] = Field(default_factory=list, description="Structured crop application & dosage rules")
    source_url: Optional[str] = Field(None, description="Authoritative source URL")
    source_type: Optional[str] = Field("OFFICIAL_NACL_WEBSITE", description="Source type provenance")
    source_date: Optional[str] = Field(None, description="Publication or revision date of source document")
    last_verified_at: Optional[str] = Field(None, description="ISO timestamp of last backend evidence verification")
    verification_status: str = Field("UNVERIFIED", description="VERIFIED, UNVERIFIED, STALE, REJECTED")


class NACLProductRecommendation(BaseModel):
    """Individual NACL product recommendation entry."""
    product_id: str = Field(..., description="Product ID slug")
    product_name: str = Field(..., description="Product brand name")
    category: str = Field(..., description="Fungicide, Insecticide, etc.")
    product_url: str = Field(..., description="Direct URL to NACL product page")
    active_ingredient: Optional[str] = Field(None, description="Active ingredient chemical formulation")
    match_type: str = Field(..., description="DIRECT_MATCH or CROP_TARGET_MATCH")
    verification_status: str = Field(..., description="VERIFIED_DIRECT_MATCH, VERIFIED_CROP_TARGET_MATCH, UNVERIFIED, REJECTED")
    dosage_verified: bool = Field(False, description="True ONLY if dosage is verified from authoritative source for exact crop+target")
    recommended_dosage: str = Field(..., description="Dosage recommendation string or 'Refer to current product label'")
    match_rationale: str = Field(..., description="Explanation of verified crop, disease/target, and authoritative source")
    verification_reason: str = Field("", description="Detailed backend audit log of validation decision")
    pack_sizes: List[str] = Field(default_factory=list, description="Commercial packaging options")
    source_url: Optional[str] = Field(None, description="Authoritative source URL")
    source_type: Optional[str] = Field(None, description="OFFICIAL_NACL_WEBSITE, OFFICIAL_LABEL, REGULATORY")
    frac_group: Optional[str] = Field(None, description="Verified FRAC code/group")


class ProductRecommendationResult(BaseModel):
    """Complete Phase 3 NACL product recommendation payload."""
    crop_name: str = Field(..., description="Target crop identified")
    disease_name: str = Field(..., description="Diagnosed disease/pathogen")
    is_healthy: bool = Field(False, description="True if plant was identified as healthy")
    recommendations: List[NACLProductRecommendation] = Field(default_factory=list, description="List of matched NACL products")
    recommended_active_ingredients: List[str] = Field(default_factory=list, description="Recommended active chemical ingredients")
    ai_advisory_summary: str = Field(..., description="Gemini AI generated treatment advisory and application timing")
    safety_disclaimer: str = Field(
        "Always follow official product label instructions, protective gear requirements, and local agricultural regulations before applying agrochemicals.",
        description="Safety disclaimer for agrochemical application"
    )
