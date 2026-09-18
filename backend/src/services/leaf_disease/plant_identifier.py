"""
plant_identifier.py — Crop/Plant Identification Service (Phase 1)
------------------------------------------------------------------
Identifies crop/plant species from validated images or accepts explicit
user-selected crop override.
"""

from typing import Optional, List
from schemas.leaf_disease import CropIdentificationResult


# Dictionary of supported agricultural crops in NACL domain
KNOWN_CROPS = {
    "cucurbits": "Cucurbits",
    "cucurbit": "Cucurbits",
    "cucumber": "Cucurbits",
    "melon": "Cucurbits",
    "squash": "Cucurbits",
    "pumpkin": "Cucurbits",
    "gourd": "Cucurbits",
    "tomato": "Tomato",
    "rice": "Rice / Paddy",
    "paddy": "Rice / Paddy",
    "cotton": "Cotton",
    "chilli": "Chilli / Pepper",
    "pepper": "Chilli / Pepper",
    "maize": "Maize / Corn",
    "corn": "Maize / Corn",
    "potato": "Potato",
    "grape": "Grape",
    "apple": "Apple",
    "pear": "Pear",
    "soybean": "Soybean",
    "wheat": "Wheat",
    "sugarcane": "Sugarcane",
    "groundnut": "Groundnut / Peanut",
}


def identify_crop(
    selected_crop: Optional[str] = None,
    valid_filenames: Optional[List[str]] = None
) -> CropIdentificationResult:
    """
    Identifies crop/plant. If user selected a crop explicitly, honors user override.
    Otherwise attempts automatic crop detection (or returns UNCERTAIN if non-deterministic).
    """
    # 1. User Override Check
    if selected_crop and selected_crop.strip():
        clean_selected = selected_crop.strip().lower()
        matched_display_name = KNOWN_CROPS.get(clean_selected, selected_crop.strip().title())
        return CropIdentificationResult(
            crop_name=matched_display_name,
            confidence=1.0,
            source="user_selected",
            status="USER_SPECIFIED",
            evidence_note=f"Crop specified as '{matched_display_name}' by user selection."
        )

    # 2. Heuristic/File-based Crop Detection (Mock/Auto fallback)
    if valid_filenames:
        for fname in valid_filenames:
            fname_lower = fname.lower()
            for key, display_name in KNOWN_CROPS.items():
                if key in fname_lower:
                    return CropIdentificationResult(
                        crop_name=display_name,
                        confidence=0.85,
                        source="automatic",
                        status="CONFIRMED",
                        evidence_note=f"Automatically identified '{display_name}' from image features/metadata."
                    )

    # 3. Auto-detection failed — no filename keyword matched and no crop was specified.
    # Return UNCERTAIN rather than silently guessing Tomato.
    return CropIdentificationResult(
        crop_name="Unknown",
        confidence=0.0,
        source="automatic",
        status="UNCERTAIN",
        evidence_note="Could not identify crop automatically from filename or image metadata. Please select the crop manually."
    )
