"""
nacl_scraper.py — NACL Agrochemical Product Catalog Scraper (Phase 2)
----------------------------------------------------------------------
Scrapes NACL Industries' official product formulations using BeautifulSoup4.
Extracts:
  - Product Name & Category (Fungicides, Insecticides, Herbicides, PGR)
  - Active Ingredient & Chemical Family
  - Mode of Action
  - Available Packaging Sizes
  - Key Benefits
  - Structured Crop Application & Dosage Rules
Exports clean JSON dataset to `nacl_catalog.json`.
"""

import re
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

from schemas.nacl_catalog import NACLProduct, CropDosageRule

logger = logging.getLogger(__name__)

# Official NACL Product Category URLs
NACL_CATEGORY_URLS = {
    "Insecticides": "https://naclind.com/products/formulations/insecticides/",
    "Fungicides": "https://naclind.com/products/formulations/fungicides/",
    "Herbicides": "https://naclind.com/products/formulations/herbicides/",
    "PGR": "https://naclind.com/products/formulations/plant-growth-regulators-and-nematicides/",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}


class NACLScraper:
    """Scrapes NACL Industries web pages and extracts structured product models."""

    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout

    def fetch_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Fetches URL content and returns a BeautifulSoup parser instance."""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return BeautifulSoup(resp.content, "html.parser")
            else:
                logger.warning("HTTP Error %d for URL: %s", resp.status_code, url)
                return None
        except Exception as e:
            logger.error("Failed to fetch URL %s: %s", url, str(e))
            return None

    def get_product_urls_from_category(self, category_url: str) -> List[str]:
        """Extracts individual product page URLs from a category listing page."""
        soup = self.fetch_soup(category_url)
        if not soup:
            return []

        urls = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # NACL product detail pages follow URL pattern: /products/<category>/<product-slug>/
            if "/products/" in href and not href.endswith("/formulations/") and href != category_url:
                # Filter out category top-level links
                if any(cat_path in href for cat_path in ["/insecticides/", "/fungicides/", "/herbicides/", "/pgr/"]):
                    # Product detail pages have a product slug after category
                    parts = [p for p in href.rstrip("/").split("/") if p]
                    if len(parts) >= 4:  # e.g. ['https:', '', 'naclind.com', 'products', 'insecticides', 'task-sc']
                        urls.add(href)

        return sorted(list(urls))

    def parse_product_page(self, product_url: str, default_category: str) -> Optional[NACLProduct]:
        """Parses an individual NACL product page into a structured NACLProduct schema."""
        soup = self.fetch_soup(product_url)
        if not soup:
            return None

        # 1. Product Slug / ID
        slug = [p for p in product_url.rstrip("/").split("/") if p][-1]
        
        # 2. Product Name
        h1 = soup.find("h1")
        product_name = h1.text.strip() if h1 else slug.replace("-", " ").title()

        # 3. Available Pack Sizes
        pack_sizes = []
        text_content = soup.get_text()
        size_match = re.search(r"Available in:\s*([^\n]+)", text_content, re.IGNORECASE)
        if size_match:
            sizes_str = size_match.group(1).strip()
            pack_sizes = [s.strip() for s in sizes_str.split(",") if s.strip()]

        # 4. Mode of Action & Active Ingredient / Chemical Family
        mode_of_action = None
        chemical_class = None
        active_ingredient = None

        # Search for Mode of Action section
        moa_header = soup.find(lambda tag: tag.name in ["h2", "h3", "h4", "strong"] and "Mode of Action" in tag.text)
        if moa_header:
            next_p = moa_header.find_next("p")
            if next_p:
                mode_of_action = next_p.text.strip()

        # Search for Category / Chemical Class line
        cat_match = re.search(r"Category:\s*([^\n]+)", text_content, re.IGNORECASE)
        if cat_match:
            chemical_class = cat_match.group(1).strip()

        # Search for active ingredient in text or subtitle
        ai_match = re.search(r"([A-Z][a-z0-9\s\%]+(?:GR|SC|EC|WP|WG|SP|SL|FS))\b", text_content)
        if ai_match:
            candidate = ai_match.group(1).strip()
            if len(candidate) > 4 and "Available" not in candidate:
                active_ingredient = candidate

        # 5. Key Benefits
        key_benefits = []
        kb_header = soup.find(lambda tag: tag.name in ["h2", "h3", "h4", "strong"] and "Key Benefits" in tag.text)
        if kb_header:
            ul = kb_header.find_next("ul")
            if ul:
                for li in ul.find_all("li"):
                    txt = li.text.strip()
                    if txt:
                        key_benefits.append(txt)

        # 6. Crop Application & Dose
        crop_rules: List[CropDosageRule] = []
        dose_header = soup.find(lambda tag: tag.name in ["h2", "h3", "h4", "strong"] and "Crop Application" in tag.text)
        if dose_header:
            # Look for <ul> or text lines following the header
            next_elem = dose_header.find_next(["ul", "p"])
            if next_elem:
                if next_elem.name == "ul":
                    for li in next_elem.find_all("li"):
                        line = li.text.strip()
                        rule = self._parse_crop_dosage_line(line)
                        if rule:
                            crop_rules.append(rule)
                else:
                    # Paragraph format
                    lines = next_elem.get_text("\n").split("\n")
                    for line in lines:
                        rule = self._parse_crop_dosage_line(line.strip())
                        if rule:
                            crop_rules.append(rule)

        # Fallback if no specific crop section was found — scan key benefits or text
        if not crop_rules:
            for benefit in key_benefits:
                if "controls the pests of" in benefit or "crops:" in benefit:
                    crops_part = re.sub(r".*controls the pests of\s*", "", benefit, flags=re.IGNORECASE)
                    crops = [c.strip() for c in crops_part.split(",") if c.strip()]
                    for c in crops:
                        crop_rules.append(CropDosageRule(crop=c.title(), target_pest_or_disease=None, dosage="As per recommendation"))

        return NACLProduct(
            product_id=slug,
            product_name=product_name,
            category=default_category,
            product_url=product_url,
            active_ingredient=active_ingredient,
            chemical_class=chemical_class,
            mode_of_action=mode_of_action,
            pack_sizes=pack_sizes,
            key_benefits=key_benefits,
            crop_applications=crop_rules
        )

    def _parse_crop_dosage_line(self, line: str) -> Optional[CropDosageRule]:
        """Parses lines like 'Rice: 400–600 ml/acre' or 'Chilli: 320–400 ml/acre' into CropDosageRule."""
        if not line or ":" not in line:
            return None
        parts = line.split(":", 1)
        crop_part = parts[0].strip()
        dosage_part = parts[1].strip()

        if len(crop_part) > 30:  # Skip non-crop headers
            return None

        return CropDosageRule(
            crop=crop_part.title(),
            target_pest_or_disease=None,
            dosage=dosage_part
        )

    def scrape_full_catalog(self) -> List[NACLProduct]:
        """Scrapes all product categories and returns complete NACLProduct list."""
        all_products: List[NACLProduct] = []

        for category, cat_url in NACL_CATEGORY_URLS.items():
            logger.info("Scraping NACL category '%s' from %s", category, cat_url)
            p_urls = self.get_product_urls_from_category(cat_url)
            logger.info("Found %d products in category '%s'", len(p_urls), category)

            for p_url in p_urls:
                time.sleep(0.3)  # Gentle pause
                prod = self.parse_product_page(p_url, category)
                if prod:
                    all_products.append(prod)
                    logger.info("  ✓ Scraped product: %s (%s)", prod.product_name, prod.category)

        return all_products


def run_catalog_scraper(output_json_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Runs NACL catalog scraper and exports JSON file."""
    if output_json_path is None:
        output_json_path = Path("src/data/nacl_catalog.json")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    scraper = NACLScraper()
    products = scraper.scrape_full_catalog()

    dict_products = [p.model_dump() for p in products]

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dict_products, f, indent=2, ensure_ascii=False)

    logger.info("Exported %d NACL products to %s", len(dict_products), str(output_json_path))
    return dict_products


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_catalog_scraper()
