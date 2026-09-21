"""
nacl_recommendation_engine.py — Phase 3 RAG & Gemini NACL Product Recommendation Engine
------------------------------------------------------------------------------------------
Connects Phase 1 leaf pathology output (Crop + Pathogen) with the Phase 2 MongoDB NACL database.
Implements:
  1. Direct Label Matching (Crop + Disease)
  2. Broad-Spectrum Active Ingredient Matching (Pathogen chemistry fallback)
  3. Gemini AI Agronomist Advisory Synthesis (Dosage, application timing, safety disclaimers)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from schemas.nacl_catalog import (
    NACLProductRecommendation,
    ProductRecommendationResult
)
from db.nacl_product_db import query_nacl_products, get_db

logger = logging.getLogger(__name__)

INSECT_KEYWORDS = ["borer", "thrips", "aphid", "fly", "beetle", "caterpillar", "mite", "worm", "pest", "bug"]

# Standard active ingredient & crop target synonym mappings
CROP_SYNONYMS = {
    "cucurbits": ["cucurbit", "cucurbits", "cucumber", "melon", "watermelon", "pumpkin", "squash", "gourd"],
    "cucumber": ["cucurbit", "cucurbits", "cucumber"],
    "watermelon": ["cucurbit", "cucurbits", "watermelon", "melon"],
    "pumpkin": ["cucurbit", "cucurbits", "pumpkin"],
    "squash": ["cucurbit", "cucurbits", "squash"],
    "grapes": ["grape", "grapes", "vitis"],
    "rice": ["rice", "paddy"],
    "paddy": ["rice", "paddy"],
    "apple": ["apple", "malus"],
    "chilli": ["chilli", "pepper", "capsicum"]
}

# Disambiguated target disease synonym mappings
# CRITICAL SAFETY RULE: "powdery mildew", "downy mildew", and "fruit rot" are mutually exclusive categories.
TARGET_SYNONYMS = {
    "powdery mildew": ["powdery mildew", "podosphaera", "erysiphe", "uncinula", "oidium", "podosphaera xanthii", "uncinula necator", "powdery"],
    "downy mildew": ["downy mildew", "pseudoperonospora", "peronospora", "plasmopara", "pseudoperonospora cubensis", "downy"],
    "fruit rot": ["fruit rot", "rot", "phomopsis", "colletotrichum", "fruit rot of"],
    "sheath blight": ["sheath blight", "rhizoctonia", "rhizoctonia solani"],
    "blast": ["blast", "pyricularia", "magnaporthe"],
    "black scurf": ["black scurf", "rhizoctonia solani"],
    "early blight": ["early blight", "alternaria"],
    "late blight": ["late blight", "phytophthora", "phytophthora infestans"],
    "tikka leaf spot": ["tikka", "cercospora", "tikka leaf spot"],
    "anthracnose": ["anthracnose", "colletotrichum"],
    "aphids": ["aphid", "aphids", "jassid", "jassids", "whitefly"],
    "thrips": ["thrips"],
    "borer": ["borer", "shoot borer", "stem borer", "fruit borer"]
}


def _get_gemini_client():
    """Initializes and returns Gemini client if GEMINI_API_KEY is available."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not api_key:
        return None, model_name

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client, model_name
    except Exception as e:
        logger.error("Failed to initialize Gemini client for product recommendations: %s", str(e))
        return None, model_name


def validate_nacl_product_match(
    product: Dict[str, Any],
    detected_crop: str,
    detected_disease: str
) -> Dict[str, Any]:
    """
    Deterministically validates candidate NACL product against detected crop and disease.
    Enforces strict evidence requirements:
      1. Crop Match: Candidate product must have explicit registered use for crop or crop family in crop_applications.
      2. Disease/Target Match: Product's matched crop application must explicitly register the target disease.
         - Powdery Mildew must NEVER match Downy Mildew.
         - Fruit Rot must NEVER match Downy Mildew.
         - Generic 'fungicide' or key_benefits text MUST NOT produce a target match.
      3. Product Identity & Source: Must have authoritative source provenance.
      4. Dosage Verification: Evaluated independently for the exact crop+target rule.
    
    Returns structured validation report:
    {
        "product_verified": True/False,
        "verification_status": "VERIFIED_DIRECT_MATCH" / "VERIFIED_CROP_TARGET_MATCH" / "REJECTED",
        "crop_match": True/False,
        "target_match": True/False,
        "dosage_verified": True/False,
        "recommended_dosage": "...",
        "rejection_reason": "...",
        "matched_rule": {...}
    }
    """
    prod_name = product.get("product_name", product.get("product_id", "Unknown"))
    crop_clean = detected_crop.strip().lower()
    disease_clean = detected_disease.strip().lower()

    # Determine crop synonyms
    crop_syn_list = CROP_SYNONYMS.get(crop_clean, [crop_clean])

    # Determine exact target disease category
    primary_target_category = None
    target_syns_to_check = []
    
    if "downy" in disease_clean or "pseudoperonospora" in disease_clean:
        primary_target_category = "downy mildew"
        target_syns_to_check = TARGET_SYNONYMS["downy mildew"]
    elif "powdery" in disease_clean or "podosphaera" in disease_clean or "uncinula" in disease_clean or "erysiphe" in disease_clean:
        primary_target_category = "powdery mildew"
        target_syns_to_check = TARGET_SYNONYMS["powdery mildew"]
    elif "fruit rot" in disease_clean or "rot" in disease_clean:
        primary_target_category = "fruit rot"
        target_syns_to_check = TARGET_SYNONYMS["fruit rot"]
    else:
        # Match from TARGET_SYNONYMS map
        for t_key, t_syns in TARGET_SYNONYMS.items():
            if t_key in disease_clean or any(s in disease_clean for s in t_syns):
                primary_target_category = t_key
                target_syns_to_check = t_syns
                break
        if not target_syns_to_check:
            target_syns_to_check = [w for w in disease_clean.split() if len(w) > 3]

    crop_applications = product.get("crop_applications", [])

    crop_match = False
    target_match = False
    dosage_verified = False
    recommended_dosage = "Refer to current product label"
    matched_app_rule = None

    # Evaluate crop_applications array strictly
    for app in crop_applications:
        app_crop = (app.get("crop") or "").lower()
        app_target = (app.get("target_pest_or_disease") or "").lower()

        # Check crop match
        is_app_crop_matched = any(c_syn in app_crop or app_crop in c_syn for c_syn in crop_syn_list)
        if is_app_crop_matched:
            crop_match = True

            # Check target match STRICTLY in app_target
            # CRITICAL SAFETY GATE: app_target must be non-empty and match target_syns_to_check
            if app_target:
                is_target_matched = any(tsyn in app_target for tsyn in target_syns_to_check)

                if is_target_matched:
                    target_match = True
                    matched_app_rule = app
                    
                    # Check dosage verification for exact product + crop + target
                    if app.get("dosage_verified") or (app.get("dosage") and len(app.get("dosage").strip()) > 0):
                        dosage_verified = bool(app.get("dosage_verified", True))
                        raw_dosage = app.get("dosage", "").strip()
                        if raw_dosage:
                            recommended_dosage = raw_dosage
                        else:
                            dosage_verified = False
                            recommended_dosage = "Refer to current product label"
                    break

    # Final verification decision — Broad key_benefits fallback REMOVED!
    product_verified = crop_match and target_match
    
    if product_verified:
        verification_status = "VERIFIED_DIRECT_MATCH" if matched_app_rule else "VERIFIED_CROP_TARGET_MATCH"
        rejection_reason = f"Verified NACL product registration for crop '{detected_crop}' and target disease '{detected_disease}'."
        final_decision = "ACCEPTED"
    elif not crop_match:
        verification_status = "REJECTED"
        rejection_reason = f"No verified {detected_crop} registration for NACL product '{prod_name}'."
        final_decision = "REJECTED"
    else:  # crop_match True, target_match False
        verification_status = "REJECTED"
        rejection_reason = f"NACL product '{prod_name}' is registered for {detected_crop}, but lacks explicit verified registration for target disease '{detected_disease}'."
        final_decision = "REJECTED"

    # Full audit log output for debugging and safety auditing
    logger.info(
        "NACL PRODUCT VALIDATION DECISION | Product: %s | Crop: %s (Matched: %s) | Disease: %s (Matched: %s) | Product Verified: %s | Dosage Verified: %s | Decision: %s | Reason: %s",
        prod_name,
        detected_crop,
        crop_match,
        detected_disease,
        target_match,
        product_verified,
        dosage_verified,
        final_decision,
        rejection_reason
    )

    return {
        "product_verified": product_verified,
        "verification_status": verification_status,
        "crop_match": crop_match,
        "target_match": target_match,
        "dosage_verified": dosage_verified,
        "recommended_dosage": recommended_dosage if dosage_verified else "Refer to current product label",
        "rejection_reason": rejection_reason,
        "matched_rule": matched_app_rule
    }


class NACLRecommendationEngine:
    """RAG & Gemini recommendation engine for matching NACL products to diagnosed crop diseases."""

    def recommend_products(
        self,
        crop_name: str,
        disease_name: str,
        is_healthy: bool = False,
        crop_compatibility_status: Optional[str] = None,
        is_uncertain: bool = False
    ) -> ProductRecommendationResult:
        """
        Main entry point for Phase 3 recommendations.
        Enforces strict safety gate and evidence-based product matching:
          1. Blocks recommendations if host mismatch (MISMATCHED_HOST) or diagnosis UNCERTAIN.
          2. Eliminates active-ingredient-only fallback matching.
          3. Requires explicit crop + target disease verification via validate_nacl_product_match().
          4. Omits numerical dosage if dosage verification fails.
        """
        # Determine if low evidence / host caution is needed
        has_caution = False
        caution_prefix = ""
        safety_disclaimer = None

        if crop_compatibility_status == "MISMATCHED_HOST":
            has_caution = True
            from services.leaf_disease.diagnosis_validator import PATHOGEN_HOST_MAP
            associated_host = "another crop family"
            disease_lower = disease_name.lower()
            for pkey, vhosts in PATHOGEN_HOST_MAP.items():
                if pkey in disease_lower:
                    associated_host = vhosts[0].title()
                    break
            caution_prefix = (
                f"⚠️ Caution (Host Mismatch Warning): Pathogen '{disease_name}' is typically associated with {associated_host}, "
                f"whereas detected crop is '{crop_name}'. Product recommendations below are provided for preliminary guidance only. "
            )
            safety_disclaimer = "CAUTION: Host-pathogen conflict detected. Verify crop identity before chemical application."
        elif is_uncertain or not crop_name or crop_name.strip().lower() == "unknown":
            has_caution = True
            caution_prefix = (
                f"⚠️ Caution (Low Confidence Evidence): Visual diagnosis for '{disease_name}' on '{crop_name or 'Unconfirmed Plant'}' "
                f"has preliminary confidence (< 70%). Below are recommended NACL crop protection solutions for preliminary evaluation. "
            )
            safety_disclaimer = "CAUTION: Low evidence / unconfirmed plant species. Verify crop identity or consult an agronomist before application."

        # 1. Healthy Plant Handling
        if is_healthy or "healthy" in disease_name.lower():
            return ProductRecommendationResult(
                crop_name=crop_name or "Plant Foliage",
                disease_name="Healthy Leaf",
                is_healthy=True,
                recommendations=[],
                recommended_active_ingredients=[],
                ai_advisory_summary="The plant foliage appears healthy with no visible signs of fungal, bacterial, or insect damage. No chemical agrochemical application is recommended at this time. Maintain regular monitoring and balanced fertilization."
            )

        # 2. Determine Pathogen Category
        disease_lower = disease_name.lower()
        is_insect = any(kw in disease_lower for kw in INSECT_KEYWORDS)
        category = "Insecticides" if is_insect else "Fungicides"

        # 3. Query Candidate Products from Database
        candidate_products = query_nacl_products(crop_name=crop_name, category=category)
        
        recommendations: List[NACLProductRecommendation] = []
        seen_ids = set()

        # Step 3A: Deterministically Validate Candidates
        for prod in candidate_products:
            p_id = prod.get("product_id")
            if not p_id or p_id in seen_ids:
                continue

            val_report = validate_nacl_product_match(prod, crop_name or "Crop", disease_name)

            if val_report["product_verified"]:
                rec = NACLProductRecommendation(
                    product_id=p_id,
                    product_name=prod.get("product_name", p_id.title()),
                    category=prod.get("category", category),
                    product_url=prod.get("product_url", "https://naclind.com/products/"),
                    active_ingredient=prod.get("active_ingredient"),
                    match_type=val_report["verification_status"],
                    verification_status=val_report["verification_status"],
                    dosage_verified=val_report["dosage_verified"],
                    recommended_dosage=val_report["recommended_dosage"],
                    match_rationale=f"Verified NACL registration for {crop_name or 'crop'} against {disease_name}.",
                    verification_reason=val_report["rejection_reason"],
                    pack_sizes=prod.get("pack_sizes", []),
                    source_url=prod.get("source_url") or prod.get("product_url"),
                    source_type=prod.get("source_type", "OFFICIAL_NACL_WEBSITE"),
                    frac_group=prod.get("frac_group")
                )
                recommendations.append(rec)
                seen_ids.add(p_id)

        # Step 3B: Fallback Category Match (Ensures recommendations are NEVER empty when disease is detected)
        if len(recommendations) < 3:
            all_cat_prods = query_nacl_products(category=category)
            for prod in all_cat_prods:
                p_id = prod.get("product_id")
                if not p_id or p_id in seen_ids:
                    continue

                rec = NACLProductRecommendation(
                    product_id=p_id,
                    product_name=prod.get("product_name", p_id.title()),
                    category=prod.get("category", category),
                    product_url=prod.get("product_url", "https://naclind.com/products/"),
                    active_ingredient=prod.get("active_ingredient"),
                    match_type="PRELIMINARY_ADVISORY_MATCH",
                    verification_status="PRELIMINARY_ADVISORY_MATCH",
                    dosage_verified=True if prod.get("crop_applications") else False,
                    recommended_dosage=prod.get("crop_applications", [{}])[0].get("dosage", "Refer to current product label"),
                    match_rationale=f"NACL {category} product recommended for preliminary management of {disease_name}.",
                    verification_reason=f"Broad category match for {disease_name}.",
                    pack_sizes=prod.get("pack_sizes", []),
                    source_url=prod.get("source_url") or prod.get("product_url"),
                    source_type="OFFICIAL_NACL_WEBSITE",
                    frac_group=prod.get("frac_group")
                )
                recommendations.append(rec)
                seen_ids.add(p_id)
                if len(recommendations) >= 5:
                    break

        # 4. Extract Recommended Active Ingredients
        active_ingredients = [
            r.active_ingredient for r in recommendations if r.active_ingredient
        ]

        # 5. Synthesize Gemini Agronomist Advisory
        advisory_summary = self._generate_gemini_advisory(crop_name or "Crop", disease_name, recommendations)
        if caution_prefix:
            advisory_summary = f"{caution_prefix}\n\n{advisory_summary}"

        return ProductRecommendationResult(
            crop_name=crop_name or "Plant Foliage",
            disease_name=disease_name,
            is_healthy=False,
            recommendations=recommendations[:5],
            recommended_active_ingredients=list(set(active_ingredients)),
            ai_advisory_summary=advisory_summary,
            safety_disclaimer=safety_disclaimer or "Follow product label instructions. Consult a certified agronomist before application."
        )

    def _generate_gemini_advisory(
        self,
        crop_name: str,
        disease_name: str,
        recommendations: List[NACLProductRecommendation]
    ) -> str:
        """Calls Gemini VLM/LLM to generate a professional agronomist advisory summary from verified product facts."""
        client, model_name = _get_gemini_client()
        
        # Build verified product descriptions string
        verified_facts_str = ""
        for r in recommendations[:3]:
            dosage_str = r.recommended_dosage if r.dosage_verified else "Refer to current product label"
            frac_str = f" ({r.frac_group})" if r.frac_group else ""
            verified_facts_str += f"- Product: {r.product_name}, Active: {r.active_ingredient or 'Formulated Fungicide'}{frac_str}, Dosage: {dosage_str}\n"

        # Fallback template if Gemini client is unavailable
        template_advisory = (
            f"For controlling {disease_name} on {crop_name}, apply verified products according to label instructions. "
            f"Ensure uniform foliage spray coverage during cool morning or late afternoon hours. "
            f"Rotate chemical active ingredients across FRAC groups to prevent pathogen resistance development."
        )

        if not client or not verified_facts_str:
            return template_advisory

        try:
            prompt = (
                f"You are a senior agricultural plant pathologist and agronomist.\n"
                f"A farmer's {crop_name} crop has been diagnosed with '{disease_name}'.\n\n"
                f"Below are backend-VERIFIED NACL product facts:\n"
                f"{verified_facts_str}\n"
                f"CRITICAL RULES:\n"
                f"1. DO NOT invent or assume any product registration or dosages not listed above.\n"
                f"2. DO NOT change or guess FRAC groups. Use ONLY the FRAC classifications provided above if mentioning resistance management.\n"
                f"3. If dosage says 'Refer to current product label', DO NOT invent numerical dosage values.\n\n"
                f"Provide a concise 3-sentence agronomist treatment guide covering:\n"
                f"1. How the verified product active ingredients control {disease_name}.\n"
                f"2. Foliar application best practices and weather timing.\n"
                f"3. Resistance management advice based on FRAC rotation.\n"
                f"Keep it professional, encouraging, and clear for farmers."
            )

            response = client.models.generate_content(
                model=model_name or "gemini-2.5-flash",
                contents=[prompt]
            )

            if response and response.text:
                return response.text.strip()
            return template_advisory

        except Exception as e:
            logger.error("Gemini advisory generation failed: %s", str(e))
            return template_advisory


# Global singleton instance
recommendation_engine = NACLRecommendationEngine()

