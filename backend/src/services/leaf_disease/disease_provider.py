"""
disease_provider.py — Disease Detection Provider Abstraction & Implementations (Phase 1)
-------------------------------------------------------------------------------------
Defines the `DiseaseDetectionProvider` interface and implementations:
  - `PlantNetDiseaseProvider`: Primary provider using Pl@ntNet `/v2/diseases/identify` (EPPO disease codes)
  - `PlantDiseaseAPIProvider`: Botanical species fallback using `/v2/identify/all`
  - `KindwiseCropHealthProvider`: Secondary provider using Kindwise crop.health API
  - `MockDiseaseDetectionProvider`: Stub/development provider for testing without credentials
"""

import os
from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Tuple
import requests

from schemas.leaf_disease import (
    NormalizedDiseaseResult,
    DiseaseCandidate,
    ImageValidationDetail,
)

logger = logging.getLogger(__name__)


class DiseaseDetectionProvider(ABC):
    """Abstract base class for disease detection providers."""

    @abstractmethod
    def detect_disease(
        self,
        crop_name: str,
        image_details: List[ImageValidationDetail],
        raw_files: Optional[List[Tuple[str, bytes]]] = None
    ) -> NormalizedDiseaseResult:
        """Invokes disease detection for validated crop and images, returning normalized result."""
        pass


class PlantDiseaseAPIProvider(DiseaseDetectionProvider):
    """
    Primary Disease API Provider reading `PLANT_API_KEY` from environment.
    Connects to Pl@ntNet API v2 endpoint for live crop and plant health identification.
    Never prints or logs the API key.
    """

    PLANTNET_BASE_URL = "https://my-api.plantnet.org/v2/identify/all"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("PLANT_API_KEY", "").strip()

    def detect_disease(
        self,
        crop_name: str,
        image_details: List[ImageValidationDetail],
        raw_files: Optional[List[Tuple[str, bytes]]] = None
    ) -> NormalizedDiseaseResult:
        if not self.api_key:
            return NormalizedDiseaseResult(
                disease_name="Provider Unavailable",
                confidence=None,
                provider_name="PlantDiseaseAPIProvider",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message="PLANT_API_KEY environment variable is not configured."
            )

        if not raw_files:
            return NormalizedDiseaseResult(
                disease_name="No Raw Image Data",
                confidence=None,
                provider_name="Pl@ntNet API (v2)",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message="No raw image bytes available for API request."
            )

        # Prepare multipart files payload for Pl@ntNet API
        files = []
        for filename, content in raw_files:
            mime = "image/jpeg"
            if filename.lower().endswith(".png"):
                mime = "image/png"
            elif filename.lower().endswith(".webp"):
                mime = "image/webp"
            files.append(("images", (filename, content, mime)))
            files.append(("organs", (None, "leaf")))

        url = f"{self.PLANTNET_BASE_URL}?api-key={self.api_key}"
        try:
            logger.info("Executing real Pl@ntNet API request for crop '%s' (%d images)", crop_name, len(raw_files))
            response = requests.post(url, files=files, timeout=15)
            
            # Safe Diagnostic Output (Status code & result length ONLY, never log API Key)
            logger.info("Pl@ntNet API HTTP Status Code: %d", response.status_code)

            if response.status_code == 200:
                res_data = response.json()
                results = res_data.get("results", [])

                if not results:
                    return NormalizedDiseaseResult(
                        disease_name="No Matching Species / Disease Found",
                        confidence=0.0,
                        provider_name="Pl@ntNet API (v2)",
                        is_mock=False,
                        supported=True,
                        candidates=[],
                        provider_status="OK",
                        error_message="API returned empty results array."
                    )

                # Parse top match species & score
                top_match = results[0]
                score = float(top_match.get("score", 0.0))
                species_info = top_match.get("species", {})
                sci_name = species_info.get("scientificNameWithoutAuthor") or species_info.get("scientificName") or "Unknown Species"
                common_names = species_info.get("commonNames") or []
                common_str = f" ({common_names[0]})" if common_names else ""
                
                disease_name = f"{sci_name}{common_str}"

                # Extract candidates list
                candidates = []
                for item in results[1:5]:
                    c_score = float(item.get("score", 0.0))
                    c_species = item.get("species", {})
                    c_sci = c_species.get("scientificNameWithoutAuthor") or c_species.get("scientificName") or "Unknown"
                    c_common = c_species.get("commonNames") or []
                    c_common_str = f" ({c_common[0]})" if c_common else ""
                    candidates.append(
                        DiseaseCandidate(disease_name=f"{c_sci}{c_common_str}", confidence=c_score)
                    )

                return NormalizedDiseaseResult(
                    disease_name=disease_name,
                    confidence=score,
                    provider_name="Pl@ntNet API (v2)",
                    is_mock=False,
                    supported=True,
                    candidates=candidates,
                    provider_status="OK",
                    error_message=None
                )

            elif response.status_code == 404:
                # Pl@ntNet returns HTTP 404 when no plant species matches the image
                return NormalizedDiseaseResult(
                    disease_name="No Matching Plant Species Found",
                    confidence=0.0,
                    provider_name="Pl@ntNet API (v2)",
                    is_mock=False,
                    supported=True,
                    candidates=[],
                    provider_status="OK",
                    error_message=None
                )

            elif response.status_code == 401:
                return NormalizedDiseaseResult(
                    disease_name="Authentication Error",
                    confidence=None,
                    provider_name="Pl@ntNet API (v2)",
                    is_mock=False,
                    supported=False,
                    candidates=[],
                    provider_status="PROVIDER_ERROR",
                    error_message="API Key Authentication Failed (HTTP 401 Unauthorized)."
                )

            else:
                return NormalizedDiseaseResult(
                    disease_name=f"API Error HTTP {response.status_code}",
                    confidence=None,
                    provider_name="Pl@ntNet API (v2)",
                    is_mock=False,
                    supported=False,
                    candidates=[],
                    provider_status="PROVIDER_ERROR",
                    error_message=f"API returned unexpected HTTP status code {response.status_code}."
                )

        except requests.exceptions.Timeout:
            logger.error("Pl@ntNet API call timed out.")
            return NormalizedDiseaseResult(
                disease_name="API Request Timeout",
                confidence=None,
                provider_name="Pl@ntNet API (v2)",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message="Connection to real plant API timed out."
            )
        except Exception as e:
            logger.error("Pl@ntNet API request failed: %s", str(e))
            return NormalizedDiseaseResult(
                disease_name="API Provider Error",
                confidence=None,
                provider_name="Pl@ntNet API (v2)",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message=f"Network or request failure: {str(e)}"
            )



class PlantNetDiseaseProvider(DiseaseDetectionProvider):
    """
    Primary Disease Detection Provider using Pl@ntNet dedicated diseases endpoint.
    Uses POST https://my-api.plantnet.org/v2/diseases/identify which returns EPPO-coded
    disease/pest candidates — actual pathology detection, not botanical species identification.
    Reads `PLANT_API_KEY` from environment. Never prints or logs the API key.
    """

    DISEASES_URL = "https://my-api.plantnet.org/v2/diseases/identify"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("PLANT_API_KEY", "").strip()

    @staticmethod
    def _format_disease_name(description: str, eppo_code: str) -> str:
        """
        Reorders Pl@ntNet disease description from: "ScientificName - Common name"
        to the user-friendly format:                "Common name (ScientificName)"
        Falls back to eppo_code if description is empty.
        """
        if not description:
            return eppo_code or "Unknown Disease"
        # Pl@ntNet format: "Plasmopara viticola - Downy mildew of grapevine"
        if " - " in description:
            parts = description.split(" - ", 1)
            scientific = parts[0].strip()
            common = parts[1].strip()
            return f"{common} ({scientific})"
        return description

    def detect_disease(
        self,
        crop_name: str,
        image_details: List[ImageValidationDetail],
        raw_files: Optional[List[Tuple[str, bytes]]] = None
    ) -> NormalizedDiseaseResult:
        if not self.api_key:
            return NormalizedDiseaseResult(
                disease_name="Provider Unavailable",
                confidence=None,
                provider_name="Pl@ntNet Diseases API",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message="PLANT_API_KEY environment variable is not configured."
            )

        if not raw_files:
            return NormalizedDiseaseResult(
                disease_name="No Raw Image Data",
                confidence=None,
                provider_name="Pl@ntNet Diseases API",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message="No raw image bytes available for API request."
            )

        # Build multipart form payload
        files = []
        for filename, content in raw_files:
            mime = "image/jpeg"
            if filename.lower().endswith(".png"):
                mime = "image/png"
            elif filename.lower().endswith(".webp"):
                mime = "image/webp"
            files.append(("images", (filename, content, mime)))
            files.append(("organs", (None, "leaf")))

        url = f"{self.DISEASES_URL}?api-key={self.api_key}&lang=en"

        try:
            logger.info("Executing Pl@ntNet diseases API request for crop '%s' (%d images)", crop_name, len(raw_files))
            response = requests.post(url, files=files, timeout=15)
            logger.info("Pl@ntNet diseases API HTTP Status Code: %d", response.status_code)

            if response.status_code == 200:
                res_data = response.json()
                results = res_data.get("results", [])

                if not results:
                    return NormalizedDiseaseResult(
                        disease_name="No Disease Candidates Found",
                        confidence=0.0,
                        provider_name="Pl@ntNet Diseases API",
                        is_mock=False,
                        supported=True,
                        candidates=[],
                        provider_status="OK",
                        error_message="API returned empty results array."
                    )

                # Parse top result — uses `description` field: "ScientificName - Common name"
                # Reformat to: "Common name (ScientificName)"
                top = results[0]
                score = float(top.get("score", 0.0))
                description = top.get("description", "")
                eppo_code = top.get("name", "")
                disease_name = self._format_disease_name(description, eppo_code)

                # Build candidate list from remaining results
                candidates = []
                for item in results[1:5]:
                    c_score = float(item.get("score", 0.0))
                    c_desc = item.get("description", "")
                    c_code = item.get("name", "")
                    c_name = self._format_disease_name(c_desc, c_code)
                    candidates.append(DiseaseCandidate(disease_name=c_name, confidence=c_score))

                return NormalizedDiseaseResult(
                    disease_name=disease_name,
                    confidence=score,
                    provider_name="Pl@ntNet Diseases API",
                    is_mock=False,
                    supported=True,
                    candidates=candidates,
                    provider_status="OK",
                    error_message=None
                )


            elif response.status_code == 404:
                # 404 = disease database has no candidates for this image
                return NormalizedDiseaseResult(
                    disease_name="No Disease Candidate Matched",
                    confidence=0.0,
                    provider_name="Pl@ntNet Diseases API",
                    is_mock=False,
                    supported=True,
                    candidates=[],
                    provider_status="OK",
                    error_message="Pl@ntNet diseases database found no matching pathogen for this image."
                )

            elif response.status_code == 401:
                return NormalizedDiseaseResult(
                    disease_name="Authentication Error",
                    confidence=None,
                    provider_name="Pl@ntNet Diseases API",
                    is_mock=False,
                    supported=False,
                    candidates=[],
                    provider_status="PROVIDER_ERROR",
                    error_message="API Key Authentication Failed (HTTP 401 Unauthorized)."
                )

            else:
                return NormalizedDiseaseResult(
                    disease_name=f"API Error HTTP {response.status_code}",
                    confidence=None,
                    provider_name="Pl@ntNet Diseases API",
                    is_mock=False,
                    supported=False,
                    candidates=[],
                    provider_status="PROVIDER_ERROR",
                    error_message=f"API returned unexpected HTTP status code {response.status_code}."
                )

        except requests.exceptions.Timeout:
            logger.error("Pl@ntNet diseases API call timed out.")
            return NormalizedDiseaseResult(
                disease_name="API Request Timeout",
                confidence=None,
                provider_name="Pl@ntNet Diseases API",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message="Connection to Pl@ntNet diseases API timed out."
            )
        except Exception as e:
            logger.error("Pl@ntNet diseases API request failed: %s", str(e))
            return NormalizedDiseaseResult(
                disease_name="API Provider Error",
                confidence=None,
                provider_name="Pl@ntNet Diseases API",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message=f"Network or request failure: {str(e)}"
            )


import base64

class KindwiseCropHealthProvider(DiseaseDetectionProvider):
    """
    Primary Disease API Provider implementing Kindwise crop.health official API contract.
    Reads `CROP_HEALTH_API_KEY` from environment.
    Identifies plant pathology, diseases, pests, nutritional deficiencies, and healthy state.
    Separates crop identification from disease/condition output.
    Never prints or logs the API key.
    """

    KINDWISE_BASE_URL = "https://crop.kindwise.com/api/v1/identification"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("CROP_HEALTH_API_KEY", "").strip()

    def detect_disease(
        self,
        crop_name: str,
        image_details: List[ImageValidationDetail],
        raw_files: Optional[List[Tuple[str, bytes]]] = None
    ) -> NormalizedDiseaseResult:
        if not self.api_key:
            return NormalizedDiseaseResult(
                disease_name="Provider Unavailable",
                confidence=None,
                provider_name="Kindwise crop.health",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message="CROP_HEALTH_API_KEY environment variable is not configured."
            )

        if not raw_files:
            return NormalizedDiseaseResult(
                disease_name="No Raw Image Data",
                confidence=None,
                provider_name="Kindwise crop.health",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message="No raw image bytes available for API request."
            )

        # Convert raw images to base64 data URIs
        b64_images = []
        for filename, content in raw_files:
            mime = "image/jpeg"
            if filename.lower().endswith(".png"):
                mime = "image/png"
            elif filename.lower().endswith(".webp"):
                mime = "image/webp"
            encoded = base64.b64encode(content).decode("utf-8")
            b64_images.append(f"data:{mime};base64,{encoded}")

        url = f"{self.KINDWISE_BASE_URL}?details=crop,health,disease,description"
        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "images": b64_images,
            "similar_images": True
        }

        try:
            logger.info("Executing Kindwise crop.health API request for crop '%s' (%d images)", crop_name, len(b64_images))
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            logger.info("Kindwise crop.health API HTTP Status Code: %d", response.status_code)

            if response.status_code in (200, 201):
                res_data = response.json()
                res_obj = res_data.get("result", {})
                
                # Health & disease predictions block
                is_healthy_obj = res_obj.get("is_healthy", {})
                is_healthy = bool(is_healthy_obj.get("binary", False))
                healthy_prob = float(is_healthy_obj.get("probability", 0.0))

                disease_obj = res_obj.get("disease", {})
                suggestions = disease_obj.get("suggestions", [])

                if is_healthy and healthy_prob > 0.7:
                    # Healthy plant condition
                    return NormalizedDiseaseResult(
                        disease_name="Healthy / No Disease Detected",
                        confidence=healthy_prob,
                        provider_name="Kindwise crop.health",
                        is_mock=False,
                        supported=True,
                        candidates=[],
                        provider_status="OK",
                        error_message=None
                    )

                if not suggestions:
                    return NormalizedDiseaseResult(
                        disease_name="No Specific Pathology / Disease Identified",
                        confidence=0.0,
                        provider_name="Kindwise crop.health",
                        is_mock=False,
                        supported=True,
                        candidates=[],
                        provider_status="OK",
                        error_message="Kindwise returned empty disease suggestions."
                    )

                # Parse top pathology / disease condition
                top_sugg = suggestions[0]
                disease_name = top_sugg.get("name", "Unknown Condition")
                confidence = float(top_sugg.get("probability", 0.0))

                candidates = []
                for item in suggestions[1:5]:
                    c_name = item.get("name", "Unknown")
                    c_prob = float(item.get("probability", 0.0))
                    candidates.append(DiseaseCandidate(disease_name=c_name, confidence=c_prob))

                return NormalizedDiseaseResult(
                    disease_name=disease_name,
                    confidence=confidence,
                    provider_name="Kindwise crop.health",
                    is_mock=False,
                    supported=True,
                    candidates=candidates,
                    provider_status="OK",
                    error_message=None
                )

            elif response.status_code == 401:
                return NormalizedDiseaseResult(
                    disease_name="Authentication Error",
                    confidence=None,
                    provider_name="Kindwise crop.health",
                    is_mock=False,
                    supported=False,
                    candidates=[],
                    provider_status="PROVIDER_ERROR",
                    error_message="API Key Authentication Failed (HTTP 401 Unauthorized)."
                )

            else:
                return NormalizedDiseaseResult(
                    disease_name=f"API Error HTTP {response.status_code}",
                    confidence=None,
                    provider_name="Kindwise crop.health",
                    is_mock=False,
                    supported=False,
                    candidates=[],
                    provider_status="PROVIDER_ERROR",
                    error_message=f"Kindwise API returned HTTP status code {response.status_code}."
                )

        except requests.exceptions.Timeout:
            logger.error("Kindwise crop.health API request timed out.")
            return NormalizedDiseaseResult(
                disease_name="API Request Timeout",
                confidence=None,
                provider_name="Kindwise crop.health",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message="Connection to Kindwise crop.health API timed out."
            )
        except Exception as e:
            logger.error("Kindwise crop.health API request failed: %s", str(e))
            return NormalizedDiseaseResult(
                disease_name="API Provider Error",
                confidence=None,
                provider_name="Kindwise crop.health",
                is_mock=False,
                supported=False,
                candidates=[],
                provider_status="PROVIDER_ERROR",
                error_message=f"Network or request failure: {str(e)}"
            )


class MockDiseaseDetectionProvider(DiseaseDetectionProvider):
    """
    Mock/Stub Disease Detection Provider for Phase 1 development and testing.
    Provides deterministic normalized results clearly flagged with `is_mock=True` and `provider_status="MOCK"`.
    """

    MOCK_DISEASE_DB = {
        "Tomato": {
            "default": ("Tomato Early Blight", 0.91, [("Septoria Leaf Spot", 0.06), ("Late Blight", 0.03)]),
            "blight": ("Tomato Early Blight", 0.89, [("Late Blight", 0.08)]),
            "spot": ("Tomato Septoria Leaf Spot", 0.88, [("Target Spot", 0.09)]),
            "healthy": ("Healthy / No Disease Detected", 0.95, []),
        },
        "Rice / Paddy": {
            "default": ("Rice Blast (Pyricularia oryzae)", 0.88, [("Brown Spot", 0.09)]),
            "blast": ("Rice Blast (Pyricularia oryzae)", 0.92, [("Sheath Blight", 0.05)]),
            "spot": ("Rice Brown Spot", 0.86, [("Bacterial Blight", 0.10)]),
        },
        "Cotton": {
            "default": ("Cotton Bacterial Blight", 0.87, [("Alternaria Leaf Spot", 0.10)]),
        },
        "Chilli / Pepper": {
            "default": ("Chilli Leaf Curl Virus", 0.90, [("Anthracnose", 0.07)]),
        },
    }

    def detect_disease(
        self,
        crop_name: str,
        image_details: List[ImageValidationDetail],
        raw_files: Optional[List[Tuple[str, bytes]]] = None
    ) -> NormalizedDiseaseResult:
        crop_entries = self.MOCK_DISEASE_DB.get(crop_name) or self.MOCK_DISEASE_DB["Tomato"]

        # Check filenames for keyword match triggers in mock mode
        matched_key = "default"
        if image_details:
            first_fname = image_details[0].filename.lower()
            if "spot" in first_fname:
                matched_key = "spot"
            elif "blight" in first_fname:
                matched_key = "blight"
            elif "healthy" in first_fname:
                matched_key = "healthy"

        disease_info = crop_entries.get(matched_key) or crop_entries["default"]
        disease_name, primary_conf, raw_candidates = disease_info

        candidates = [
            DiseaseCandidate(disease_name=name, confidence=conf)
            for name, conf in raw_candidates
        ]

        return NormalizedDiseaseResult(
            disease_name=disease_name,
            confidence=primary_conf,
            provider_name="MockDiseaseDetectionProvider",
            is_mock=True,
            supported=True,
            candidates=candidates,
            provider_status="MOCK",
            error_message=None
        )


def get_disease_provider() -> DiseaseDetectionProvider:
    """
    Factory function returning the active disease detection provider.
    Priority order:
    1. If `PLANT_API_KEY` is set, returns `PlantNetDiseaseProvider` (dedicated /v2/diseases/identify — EPPO pathology).
    2. Else if `CROP_HEALTH_API_KEY` is set, returns `KindwiseCropHealthProvider`.
    3. Else defaults to `MockDiseaseDetectionProvider`.
    """
    plant_key = os.getenv("PLANT_API_KEY", "").strip()
    if plant_key:
        return PlantNetDiseaseProvider(api_key=plant_key)
    crop_key = os.getenv("CROP_HEALTH_API_KEY", "").strip()
    if crop_key:
        return KindwiseCropHealthProvider(api_key=crop_key)
    return MockDiseaseDetectionProvider()

