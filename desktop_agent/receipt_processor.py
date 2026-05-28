"""
Smart Cowork Receipt Processor — 6-Phase Agentic Pipeline.

Processes a folder of receipt images through:
  1. SCAN    — inventory images, estimate batches
  2. OCR     — extract text via llava:7b (Ollama vision API)
  3. CLASSIFY— categorize via qwen3:8b
  4. ORGANIZE— create subfolders, move/copy files
  5. REPORT  — generate formatted .xlsx spreadsheet
  6. VERIFY  — spot-check OCR accuracy with llama3.2

Uses 4 local Ollama LLMs — zero external APIs.
"""

import os
import re
import json
import time
import base64
import shutil
import random
import logging
import requests
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OCR_MODEL = os.getenv("OCR_MODEL", "llava:7b")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "qwen3:8b")
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "llama3.2:latest")
REPORTER_MODEL = os.getenv("REPORTER_MODEL", "qwen3:4b")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
BATCH_SIZE = 10
BATCH_PAUSE_SEC = 1.5
MAX_RETRIES = 2

STANDARD_CATEGORIES = [
    "Food_Dining", "Gas_Fuel", "Transport", "Travel_Lodging",
    "Grocery_Retail", "Software_SaaS", "Entertainment",
    "Utilities", "Healthcare", "Office_Supplies", "Uncategorized",
]


# ── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class ReceiptData:
    """Structured OCR extraction from a single receipt."""
    filename: str = ""
    vendor: str = ""
    date: str = ""                  # YYYY-MM-DD normalized
    amount: float = 0.0
    currency: str = "USD"
    category: str = "Uncategorized"
    items: List[dict] = field(default_factory=list)
    payment_method: str = ""
    receipt_number: str = ""
    tax: float = 0.0
    subtotal: float = 0.0
    ocr_confidence: float = 0.0     # 0.0-1.0
    ocr_raw_text: str = ""
    original_path: str = ""
    organized_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessingProgress:
    """Real-time progress tracking."""
    phase: str = "idle"
    phase_num: int = 0
    total_phases: int = 6
    batch_current: int = 0
    batch_total: int = 0
    files_processed: int = 0
    files_total: int = 0
    current_file: str = ""
    errors: List[str] = field(default_factory=list)
    start_time: float = 0.0

    def status_line(self) -> str:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return (f"Phase {self.phase_num}/{self.total_phases} — {self.phase} — "
                f"{self.files_processed}/{self.files_total} files — {elapsed:.0f}s")


# ── Ollama Helpers ──────────────────────────────────────────────────────────

def _resize_image_b64(image_path: str, max_dim: int = 768) -> str:
    """Resize image and return base64-encoded JPEG.

    768px is a good balance for llava:7b — enough detail for OCR
    while keeping payloads manageable.
    """
    try:
        from PIL import Image
        import io
        img = Image.open(image_path)
        # Resize if larger than max_dim
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        # Convert to RGB (drop alpha channel)
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        logger.warning("Pillow not installed — sending raw image (may be large)")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Image resize failed, using raw: {e}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# Vision options — keep minimal for broad model compatibility (moondream, llava, etc.)
_VISION_OPTIONS = {
    "temperature": 0.1,
    "num_predict": 1024,
}

def _call_ollama_vision(image_path: str, prompt: str,
                        model: str = OCR_MODEL, timeout: int = 90) -> str:
    """Call Ollama vision API with a base64-encoded image.

    Strategy:
      1. Resize image to max 512px
      2. Try /api/generate first (more compatible with vision models)
      3. Fallback to /api/chat if /api/generate fails
    """
    img_b64 = _resize_image_b64(image_path)
    fname = Path(image_path).name

    # ── Attempt 1: /api/generate (preferred for vision models) ──
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": _VISION_OPTIONS,
        }, timeout=timeout)
        if r.status_code == 200:
            resp_json = r.json()
            content = resp_json.get("response", "")
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if content:
                logger.info(f"[OCR] /api/generate OK for {fname}: {content[:80]}...")
                return content
            # Log raw response for debugging empty content
            logger.warning(f"[OCR] /api/generate returned empty for {fname}. "
                           f"Raw keys: {list(resp_json.keys())}, "
                           f"done: {resp_json.get('done')}, "
                           f"eval_count: {resp_json.get('eval_count', 'N/A')}")
        else:
            body = r.text[:300] if r.text else "no body"
            logger.warning(f"[OCR] /api/generate returned {r.status_code}: {body}")
    except requests.exceptions.Timeout:
        logger.warning(f"[OCR] /api/generate timed out for {fname}")
    except Exception as e:
        logger.warning(f"[OCR] /api/generate failed: {e}")

    # ── Attempt 2: /api/chat (fallback) ──
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }],
            "stream": False,
            "options": _VISION_OPTIONS,
        }, timeout=timeout)
        if r.status_code == 200:
            resp_json = r.json()
            content = resp_json.get("message", {}).get("content", "")
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if content:
                logger.info(f"[OCR] /api/chat OK for {fname}: {content[:80]}...")
                return content
            logger.warning(f"[OCR] /api/chat also empty for {fname}. "
                           f"Raw keys: {list(resp_json.keys())}, "
                           f"done: {resp_json.get('done')}")
            return ""
        else:
            body = r.text[:300] if r.text else "no body"
            logger.error(f"[OCR] /api/chat failed {r.status_code}: {body}")
            return f"[ERROR] {model} returned {r.status_code}"
    except requests.exceptions.Timeout:
        return f"[TIMEOUT] {model} timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] {model} failed: {e}"


def _call_ollama_text(prompt: str, system: str = "",
                      model: str = CLASSIFIER_MODEL,
                      timeout: int = 60, max_tokens: int = 2048) -> str:
    """Call Ollama text API (non-vision)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json={
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.2, "num_predict": max_tokens, "repeat_penalty": 1.3},
        }, timeout=timeout)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    except requests.exceptions.Timeout:
        return f"[TIMEOUT] {model} timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] {model} failed: {e}"


def _extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM output."""
    if not isinstance(text, str):
        return {}
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting from code blocks
    patterns = [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'(\{[^{}]*\})', r'(\{.*\})']
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                continue
    return {}


# ── Receipt Processor ───────────────────────────────────────────────────────

class ReceiptProcessor:
    """6-phase agentic receipt processing pipeline."""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        self.progress = ProcessingProgress()
        self.receipts: List[ReceiptData] = []
        self.audit_log: List[dict] = []
        logger.info(f"ReceiptProcessor initialized (base={self.base_path})")

    def _log(self, phase: str, action: str, details: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "action": action,
            "details": details,
        }
        self.audit_log.append(entry)
        logger.info(f"[{phase}] {action}: {details}")

    # ── Phase 1: SCAN ───────────────────────────────────────────────────────

    def scan_folder(self, folder_path: str) -> dict:
        """Phase 1: Inventory all receipt images in the folder."""
        self.progress = ProcessingProgress(
            phase="SCAN", phase_num=1, start_time=time.time()
        )
        folder = Path(folder_path)
        if not folder.is_absolute():
            folder = self.base_path / folder
        if not folder.exists():
            return {"error": f"Folder not found: {folder}", "images": []}

        images = sorted([
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        self.progress.files_total = len(images)
        batch_count = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE

        self._log("SCAN", f"Found {len(images)} receipt images",
                   f"batches={batch_count}, folder={folder}")

        return {
            "folder": str(folder),
            "image_count": len(images),
            "images": [str(p) for p in images],
            "batch_count": batch_count,
            "batch_size": BATCH_SIZE,
            "total_size_mb": round(sum(f.stat().st_size for f in images) / 1024 / 1024, 2),
            "options": {
                "org_method": ["move", "copy", "report_only"],
                "category_logic": ["auto_detect", "standard_business", "personal_budget"],
            },
        }

    # ── Phase 2: OCR ────────────────────────────────────────────────────────

    def ocr_single(self, image_path: str) -> ReceiptData:
        """OCR a single receipt image.

        Strategy A: Simple prompt → get raw text → parse with regex
        Strategy B: If raw text has enough data, try JSON prompt on text model
        """
        self.progress.current_file = Path(image_path).name

        receipt = ReceiptData(
            filename=Path(image_path).name,
            original_path=str(image_path),
        )

        # ── Strategy A: Simple natural language prompt ──
        simple_prompt = (
            "Read this receipt image carefully. List the following details:\n"
            "Store name:\n"
            "Date:\n"
            "Total amount:\n"
            "Tax:\n"
            "Payment method:\n"
            "Items purchased with prices:\n"
            "Receipt number:"
        )

        raw = _call_ollama_vision(image_path, simple_prompt, model=OCR_MODEL, timeout=60)
        receipt.ocr_raw_text = raw

        if raw and not raw.startswith("[ERROR]") and not raw.startswith("[TIMEOUT]"):
            # Parse the natural language response
            parsed = self._parse_ocr_text(raw)
            # Accept if we got ANY useful data (vendor, amount, or date)
            has_data = parsed.get("vendor") or parsed.get("amount") or parsed.get("date")
            if has_data:
                receipt.vendor = parsed.get("vendor", "")
                receipt.date = self._normalize_date(parsed.get("date", ""))
                receipt.amount = self._parse_amount(parsed.get("amount", 0))
                receipt.tax = self._parse_amount(parsed.get("tax", 0))
                receipt.subtotal = self._parse_amount(parsed.get("subtotal", 0))
                receipt.payment_method = parsed.get("payment_method", "")
                receipt.receipt_number = parsed.get("receipt_number", "")
                receipt.items = parsed.get("items", [])
                receipt.category = parsed.get("category_hint", "Uncategorized")

                # Confidence scoring
                score = 0.0
                if receipt.vendor:
                    score += 0.35
                if receipt.amount > 0:
                    score += 0.35
                if receipt.date:
                    score += 0.30
                receipt.ocr_confidence = round(score, 2)

                self._log("OCR", f"Extracted {receipt.filename}",
                           f"vendor={receipt.vendor}, amount=${receipt.amount:.2f}, "
                           f"date={receipt.date}, confidence={receipt.ocr_confidence}")
                return receipt

        # ── Strategy B: Try JSON prompt as fallback ──
        json_prompt = "What store, date, and total dollar amount is on this receipt? Reply as: Store: ... Date: ... Total: $..."
        raw2 = _call_ollama_vision(image_path, json_prompt, model=OCR_MODEL, timeout=45)
        if raw2 and not raw2.startswith("[ERROR]"):
            receipt.ocr_raw_text = raw2
            parsed = self._parse_ocr_text(raw2)
            receipt.vendor = parsed.get("vendor", "")
            receipt.date = self._normalize_date(parsed.get("date", ""))
            receipt.amount = self._parse_amount(parsed.get("amount", 0))
            receipt.payment_method = parsed.get("payment_method", "")

            score = 0.0
            if receipt.vendor:
                score += 0.35
            if receipt.amount > 0:
                score += 0.35
            if receipt.date:
                score += 0.30
            receipt.ocr_confidence = round(score, 2)
        else:
            receipt.ocr_confidence = 0.0
            self._log("OCR", f"Both strategies failed for {receipt.filename}", raw[:200])

        return receipt

    @staticmethod
    def _parse_ocr_text(text: str) -> dict:
        """Parse natural language OCR output into structured data using regex."""
        result = {"vendor": "", "date": "", "amount": 0, "tax": 0, "subtotal": 0,
                  "payment_method": "", "receipt_number": "", "items": [], "category_hint": ""}

        if not text:
            return result

        lines = text.strip().split("\n")
        text_lower = text.lower()

        # ── Vendor: look for "Store name:" or labeled vendor ──
        for line in lines:
            m = re.search(r'(?:store\s*(?:name)?|vendor|merchant|company)\s*[:=]\s*(.+)', line, re.I)
            if m:
                result["vendor"] = m.group(1).strip().strip('"').strip("'")
                break
        if not result["vendor"]:
            # Try lines, stripping leading numbers like "1. Vendor Name"
            for line in lines:
                cleaned = line.strip().strip('-').strip('*').strip()
                # Strip numbered list prefix: "1. " or "1) "
                cleaned = re.sub(r'^\d+[.)\s]+', '', cleaned).strip()
                if cleaned and len(cleaned) > 2 and not re.match(
                    r'^(date|total|tax|item|receipt|payment|read|store|here|the following|list)',
                    cleaned, re.I
                ):
                    result["vendor"] = cleaned
                    break

        # Clean vendor: truncate at dollar sign, semicolon; cap at 40 chars
        if result["vendor"]:
            v = result["vendor"]
            for sep in ['$', ';', ',  ', ' - ']:
                if sep in v:
                    v = v[:v.index(sep)]
            v = v.strip().rstrip('.,;:-')
            result["vendor"] = v[:40].strip()

        # ── Date: find MM/DD/YYYY, YYYY-MM-DD, or labeled date ──
        date_patterns = [
            r'(?:date)\s*[:=]\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            r'(?:date)\s*[:=]\s*(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}-\d{1,2}-\d{4})',
        ]
        for pattern in date_patterns:
            m = re.search(pattern, text, re.I)
            if m:
                result["date"] = m.group(1).strip()
                break

        # ── Amount: find "Total: $X.XX" or largest dollar amount ──
        total_patterns = [
            r'(?:total\s*(?:amount|charged)?)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)',
            r'(?:grand\s*total)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)',
            r'(?:amount\s*(?:due|paid)?)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)',
        ]
        for pattern in total_patterns:
            m = re.search(pattern, text, re.I)
            if m:
                result["amount"] = m.group(1).replace(",", "")
                break
        if not result["amount"]:
            # Find all dollar amounts and take the largest
            amounts = re.findall(r'\$\s*([\d,]+\.\d{2})', text)
            if amounts:
                result["amount"] = max(amounts, key=lambda x: float(x.replace(",", "")))

        # ── Tax ──
        m = re.search(r'(?:tax)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)', text, re.I)
        if m:
            result["tax"] = m.group(1).replace(",", "")

        # ── Subtotal ──
        m = re.search(r'(?:subtotal|sub-total|sub total)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)', text, re.I)
        if m:
            result["subtotal"] = m.group(1).replace(",", "")

        # ── Payment Method ──
        m = re.search(r'(?:payment\s*(?:method|type)?)\s*[:=]\s*(.+)', text, re.I)
        if m:
            result["payment_method"] = m.group(1).strip()
        elif re.search(r'\b(visa|mastercard|amex|cash|debit|credit|mc\s*\*{4})\b', text, re.I):
            m = re.search(r'\b(visa|mastercard|amex|cash|debit|credit|MC\s*\*{4}\d*)\b', text, re.I)
            if m:
                result["payment_method"] = m.group(1).strip()

        # ── Receipt Number ──
        m = re.search(r'(?:receipt\s*(?:#|number|no)?|invoice\s*(?:#|number)?|trans(?:action)?\s*#?)\s*[:=]?\s*([A-Z0-9\-]+)', text, re.I)
        if m:
            result["receipt_number"] = m.group(1).strip()

        # ── Items: look for "item: $price" patterns ──
        item_patterns = re.findall(r'[-*]\s*(.+?)\s*[\-:=]?\s*\$?\s*(\d+\.?\d*)', text)
        if item_patterns:
            result["items"] = [{"name": name.strip(), "price": float(price)} for name, price in item_patterns[:10]]

        # ── Category hint from vendor/text ──
        cat_keywords = {
            "gas": "gas", "fuel": "gas", "shell": "gas", "exxon": "gas", "chevron": "gas",
            "restaurant": "food", "cafe": "food", "coffee": "food", "pizza": "food", "grill": "food",
            "costco": "retail", "walmart": "retail", "target": "retail", "grocery": "retail",
            "uber": "transport", "lyft": "transport", "lime": "transport", "scooter": "transport",
            "enterprise": "transport", "carshare": "transport", "car rental": "transport",
            "hotel": "travel", "airbnb": "travel", "flight": "travel",
            "slack": "software", "github": "software", "subscription": "software", "saas": "software",
            "netflix": "entertainment", "spotify": "entertainment",
        }
        for keyword, cat in cat_keywords.items():
            if keyword in text_lower:
                result["category_hint"] = cat
                break

        return result

    def ocr_batch(self, image_paths: List[str], batch_size: int = BATCH_SIZE) -> List[ReceiptData]:
        """Phase 2: OCR all images in batches with retry."""
        self.progress.phase = "OCR"
        self.progress.phase_num = 2
        self.progress.files_total = len(image_paths)
        self.progress.batch_total = (len(image_paths) + batch_size - 1) // batch_size

        results = []
        for batch_idx in range(0, len(image_paths), batch_size):
            batch = image_paths[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            self.progress.batch_current = batch_num
            self._log("OCR", f"Batch {batch_num}/{self.progress.batch_total}",
                       f"{len(batch)} images")

            for img_path in batch:
                retries = 0
                receipt = None
                while retries <= MAX_RETRIES:
                    try:
                        receipt = self.ocr_single(img_path)
                        if receipt.ocr_confidence > 0:
                            break
                        retries += 1
                        if retries <= MAX_RETRIES:
                            self._log("OCR", f"Retry {retries}/{MAX_RETRIES}",
                                       Path(img_path).name)
                            time.sleep(0.5)
                    except Exception as e:
                        retries += 1
                        self._log("OCR", f"Error on {Path(img_path).name}", str(e))
                        receipt = ReceiptData(
                            filename=Path(img_path).name,
                            original_path=str(img_path),
                            ocr_confidence=0.0,
                            ocr_raw_text=f"[ERROR] {e}",
                        )

                if receipt:
                    results.append(receipt)
                self.progress.files_processed += 1

            # Pause between batches to respect GPU memory
            if batch_idx + batch_size < len(image_paths):
                time.sleep(BATCH_PAUSE_SEC)

        self._log("OCR", f"Completed {len(results)} receipts",
                   f"avg_confidence={sum(r.ocr_confidence for r in results)/max(len(results),1):.2f}")
        return results

    # ── Phase 3: CLASSIFY ───────────────────────────────────────────────────

    def classify_receipts(self, receipts: List[ReceiptData]) -> List[ReceiptData]:
        """Phase 3: Normalize categories using qwen3:8b."""
        self.progress.phase = "CLASSIFY"
        self.progress.phase_num = 3

        # Build summary for LLM classification
        items_summary = []
        for i, r in enumerate(receipts):
            items_summary.append(
                f"{i}: vendor=\"{r.vendor}\", hint=\"{r.category}\", "
                f"amount=${r.amount:.2f}, items={[it.get('name','') for it in r.items[:3]]}"
            )
        items_text = "\n".join(items_summary)

        prompt = f"""Classify each receipt into EXACTLY one of these categories:
{json.dumps(STANDARD_CATEGORIES)}

Receipt list:
{items_text}

Return a JSON array of objects, one per receipt, in the same order:
[{{"index": 0, "category": "Gas_Fuel"}}, {{"index": 1, "category": "Food_Dining"}}, ...]

Rules:
- Gas stations, fuel → Gas_Fuel
- Restaurants, cafes, coffee shops, food delivery → Food_Dining
- Grocery stores (Costco, Walmart, HEB) → Grocery_Retail
- Uber, Lyft, Lime, scooters, car rentals → Transport
- Hotels, Airbnb, flights → Travel_Lodging
- SaaS subscriptions, software licenses → Software_SaaS
- Movies, streaming, games → Entertainment
- Electric, water, internet, phone → Utilities
- Pharmacy, doctor, dental → Healthcare
- Stationery, printer, office furniture → Office_Supplies
- If unsure → Uncategorized

Return ONLY the JSON array."""

        raw = _call_ollama_text(prompt, model=CLASSIFIER_MODEL, timeout=90, max_tokens=4096)

        # Parse classification results
        try:
            # Try to extract JSON array
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                classifications = json.loads(match.group())
            else:
                classifications = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            self._log("CLASSIFY", "LLM classification parse failed, using category hints")
            classifications = []

        # Apply classifications
        cat_map = {c.get("index", -1): c.get("category", "Uncategorized") for c in classifications}
        for i, r in enumerate(receipts):
            if i in cat_map and cat_map[i] in STANDARD_CATEGORIES:
                r.category = cat_map[i]
            else:
                # Fallback: use the category hint from OCR or keyword matching
                r.category = self._keyword_category(r)

        cat_counts = {}
        for r in receipts:
            cat_counts[r.category] = cat_counts.get(r.category, 0) + 1
        self._log("CLASSIFY", f"Classified {len(receipts)} receipts", str(cat_counts))

        return receipts

    def _keyword_category(self, receipt: ReceiptData) -> str:
        """Fallback keyword-based categorization using vendor + raw OCR text."""
        # Search across vendor name, category hint, items, AND raw OCR text
        text = f"{receipt.vendor} {receipt.category} {receipt.ocr_raw_text} {' '.join(it.get('name','') for it in receipt.items)}".lower()
        mappings = {
            "Gas_Fuel": ["shell", "exxon", "chevron", "bp ", "gas", "fuel", "gallons", "pump", "valero", "circle k fuel", "quicktrip", "racetrac", "texaco", "citgo"],
            "Food_Dining": ["restaurant", "cafe", "coffee", "pizza", "burger", "sushi", "dine", "grill", "kitchen", "taco", "starbucks", "mcdonald", "breakfast", "bistro", "steakhouse", "ribeye", "boulange", "golden fork", "fiesta"],
            "Grocery_Retail": ["costco", "walmart", "target", "heb", "kroger", "safeway", "grocery", "wholesale", "trader joe", "nordstrom", "rei ", "home depot", "best buy"],
            "Transport": ["uber", "lyft", "lime", "scooter", "e-scooter", "e-bike", "taxi", "enterprise", "carshare", "car rental", "transit", "metro", "bus", "trip receipt", "bergstrom"],
            "Travel_Lodging": ["hotel", "airbnb", "marriott", "hilton", "flight", "airline", "expedia", "booking", "embassy suites", "residence inn", "garden inn", "folding table", "room charge", "check-in", "room service", "the line hotel"],
            "Software_SaaS": ["slack", "github", "aws", "azure", "google cloud", "dropbox", "adobe", "microsoft", "subscription", "saas", "notion", "anthropic", "amazon business", "invoice"],
            "Entertainment": ["netflix", "spotify", "hulu", "movie", "theater", "cinema", "gaming", "steam", "lift ticket"],
            "Utilities": ["electric", "water", "gas bill", "internet", "phone", "verizon", "at&t", "comcast"],
            "Healthcare": ["pharmacy", "cvs health", "walgreens", "doctor", "dental", "medical", "hospital"],
            "Office_Supplies": ["staples", "office depot", "printer", "paper", "stationery"],
        }
        for category, keywords in mappings.items():
            if any(kw in text for kw in keywords):
                return category
        return "Uncategorized"

    # ── Phase 4: ORGANIZE ───────────────────────────────────────────────────

    def organize_files(self, receipts: List[ReceiptData],
                       output_dir: str = None,
                       method: str = "move") -> dict:
        """Phase 4: Create category folders and move files directly into them.

        Files are MOVED into category subfolders directly inside the receipts folder.
        After processing, the receipts folder only contains category subfolders.
        """
        self.progress.phase = "ORGANIZE"
        self.progress.phase_num = 4

        if output_dir is None:
            # Create category folders directly inside the receipts folder (no 'organized' subfolder)
            first_receipt = receipts[0] if receipts else None
            if first_receipt and first_receipt.original_path:
                output_dir = str(Path(first_receipt.original_path).parent)
            else:
                output_dir = str(self.base_path)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {"moved": 0, "errors": 0, "folders_created": set()}

        for r in receipts:
            if not r.original_path or not Path(r.original_path).exists():
                stats["errors"] += 1
                continue

            # Create category subfolder
            cat_folder = output_path / r.category
            cat_folder.mkdir(exist_ok=True)
            stats["folders_created"].add(r.category)

            # Build descriptive filename: YYYY-MM-DD_Vendor_$Amount.ext
            vendor_clean = re.sub(r'[^\w\s-]', '', r.vendor)[:20].strip().replace(' ', '_')
            # Sanitize date: replace any slashes with dashes to prevent path issues
            date_str = r.date.replace('/', '-') if r.date else "undated"
            ext = Path(r.original_path).suffix
            new_name = f"{date_str}_{vendor_clean}_${r.amount:.2f}{ext}"
            # Extra safety: remove any remaining path separators in filename
            new_name = new_name.replace('/', '-').replace('\\', '-')
            dest = cat_folder / new_name

            # Handle name collisions
            counter = 1
            while dest.exists():
                stem = f"{date_str}_{vendor_clean}_${r.amount:.2f}_{counter}"
                dest = cat_folder / f"{stem}{ext}"
                counter += 1

            try:
                # Default to move; only explicit "copy" does copy
                if method == "copy":
                    shutil.copy2(str(r.original_path), str(dest))
                else:
                    shutil.move(str(r.original_path), str(dest))
                r.organized_path = str(dest)
                stats["moved"] += 1
                self._log("ORGANIZE", f"move {r.filename}", str(dest))
            except Exception as e:
                stats["errors"] += 1
                self._log("ORGANIZE", f"Error on {r.filename}", str(e))

        stats["folders_created"] = list(stats["folders_created"])
        self._log("ORGANIZE", f"Organized {stats['moved']} files into "
                   f"{len(stats['folders_created'])} folders")
        return stats

    # ── Phase 5: REPORT ─────────────────────────────────────────────────────

    def generate_report(self, receipts: List[ReceiptData],
                        output_path: str = None) -> str:
        """Phase 5: Generate formatted .xlsx spreadsheet."""
        self.progress.phase = "REPORT"
        self.progress.phase_num = 5

        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
        except ImportError:
            self._log("REPORT", "openpyxl not installed, generating CSV fallback")
            return self._generate_csv_fallback(receipts, output_path)

        if output_path is None:
            if receipts and receipts[0].original_path:
                parent = Path(receipts[0].original_path).parent
            else:
                parent = self.base_path
            output_path = str(parent / "receipt_summary.xlsx")

        wb = Workbook()

        # ── Styles ──
        header_font = Font(bold=True, color="FFFFFF", size=11)
        header_fill = PatternFill(start_color="2D5F8A", end_color="2D5F8A", fill_type="solid")
        money_fmt = '#,##0.00'
        pct_fmt = '0.0%'
        thin_border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
        )
        alt_fill = PatternFill(start_color="F2F7FB", end_color="F2F7FB", fill_type="solid")

        def style_header(ws, num_cols):
            for col in range(1, num_cols + 1):
                cell = ws.cell(row=1, column=col)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")
                cell.border = thin_border

        def auto_width(ws):
            for col_cells in ws.columns:
                max_len = 0
                col_letter = get_column_letter(col_cells[0].column)
                for cell in col_cells:
                    val = str(cell.value or "")
                    max_len = max(max_len, len(val))
                ws.column_dimensions[col_letter].width = min(max_len + 3, 40)

        # ── Sheet 1: All Receipts ──
        ws1 = wb.active
        ws1.title = "All_Receipts"
        headers = ["#", "Date", "Vendor", "Category", "Amount", "Tax", "Subtotal",
                    "Currency", "Payment Method", "Items", "Receipt #",
                    "OCR Confidence", "Original File", "Organized Path", "Raw OCR Text"]
        ws1.append(headers)
        style_header(ws1, len(headers))

        for i, r in enumerate(sorted(receipts, key=lambda x: x.date or "9999")):
            items_str = "; ".join(f"{it.get('name','')} ${it.get('price',0)}" for it in r.items[:5])
            # Get just the relative organized path (category/filename)
            org_rel = ""
            if r.organized_path:
                parts = Path(r.organized_path).parts
                org_rel = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1] if parts else ""
            # Truncate raw OCR text to 200 chars for readability
            ocr_snippet = (r.ocr_raw_text[:200] + "...") if len(r.ocr_raw_text) > 200 else r.ocr_raw_text
            row = [
                i + 1, r.date, r.vendor, r.category, r.amount, r.tax, r.subtotal,
                r.currency, r.payment_method, items_str, r.receipt_number,
                r.ocr_confidence, r.filename, org_rel, ocr_snippet,
            ]
            ws1.append(row)
            row_num = i + 2
            # Money formatting (columns shifted by 1 due to # column)
            for col in [5, 6, 7]:  # Amount, Tax, Subtotal
                ws1.cell(row=row_num, column=col).number_format = money_fmt
            # Confidence as percentage
            ws1.cell(row=row_num, column=12).number_format = pct_fmt
            # Alternating row colors
            if i % 2 == 0:
                for col in range(1, len(headers) + 1):
                    ws1.cell(row=row_num, column=col).fill = alt_fill
            # Borders
            for col in range(1, len(headers) + 1):
                ws1.cell(row=row_num, column=col).border = thin_border

        # Totals row for Sheet 1
        totals_row_num = len(receipts) + 2
        ws1.cell(row=totals_row_num, column=1).value = ""
        ws1.cell(row=totals_row_num, column=2).value = "TOTALS"
        ws1.cell(row=totals_row_num, column=3).value = f"{len(receipts)} receipts"
        ws1.cell(row=totals_row_num, column=5).value = sum(r.amount for r in receipts)
        ws1.cell(row=totals_row_num, column=5).number_format = money_fmt
        ws1.cell(row=totals_row_num, column=6).value = sum(r.tax for r in receipts)
        ws1.cell(row=totals_row_num, column=6).number_format = money_fmt
        for col in range(1, len(headers) + 1):
            cell = ws1.cell(row=totals_row_num, column=col)
            cell.font = Font(bold=True)
            cell.border = thin_border

        auto_width(ws1)

        # ── Sheet 2: Category Summary ──
        ws2 = wb.create_sheet("Category_Summary")
        cat_headers = ["Category", "Count", "Total Amount", "Avg Amount", "% of Total"]
        ws2.append(cat_headers)
        style_header(ws2, len(cat_headers))

        cat_stats = {}
        grand_total = sum(r.amount for r in receipts)
        for r in receipts:
            if r.category not in cat_stats:
                cat_stats[r.category] = {"count": 0, "total": 0.0}
            cat_stats[r.category]["count"] += 1
            cat_stats[r.category]["total"] += r.amount

        for i, (cat, st) in enumerate(sorted(cat_stats.items(), key=lambda x: -x[1]["total"])):
            avg = st["total"] / max(st["count"], 1)
            pct = st["total"] / max(grand_total, 0.01)
            ws2.append([cat, st["count"], st["total"], avg, pct])
            row_num = i + 2
            ws2.cell(row=row_num, column=3).number_format = money_fmt
            ws2.cell(row=row_num, column=4).number_format = money_fmt
            ws2.cell(row=row_num, column=5).number_format = pct_fmt
            if i % 2 == 0:
                for col in range(1, len(cat_headers) + 1):
                    ws2.cell(row=row_num, column=col).fill = alt_fill
            for col in range(1, len(cat_headers) + 1):
                ws2.cell(row=row_num, column=col).border = thin_border

        # Totals row
        total_row = ["TOTAL", len(receipts), grand_total, grand_total / max(len(receipts), 1), 1.0]
        ws2.append(total_row)
        total_row_num = len(cat_stats) + 2
        for col in range(1, len(cat_headers) + 1):
            cell = ws2.cell(row=total_row_num, column=col)
            cell.font = Font(bold=True)
            cell.border = thin_border
        ws2.cell(row=total_row_num, column=3).number_format = money_fmt
        ws2.cell(row=total_row_num, column=4).number_format = money_fmt
        ws2.cell(row=total_row_num, column=5).number_format = pct_fmt

        auto_width(ws2)

        # ── Sheet 3: Monthly Breakdown ──
        ws3 = wb.create_sheet("Monthly_Breakdown")

        # Build monthly pivot
        monthly = {}  # {YYYY-MM: {category: amount}}
        all_cats = sorted(cat_stats.keys())
        for r in receipts:
            month = r.date[:7] if r.date and len(r.date) >= 7 else "Unknown"
            if month not in monthly:
                monthly[month] = {c: 0.0 for c in all_cats}
            if r.category in monthly[month]:
                monthly[month][r.category] += r.amount

        month_headers = ["Month"] + all_cats + ["Total"]
        ws3.append(month_headers)
        style_header(ws3, len(month_headers))

        for i, (month, cats) in enumerate(sorted(monthly.items())):
            row = [month] + [cats.get(c, 0.0) for c in all_cats] + [sum(cats.values())]
            ws3.append(row)
            row_num = i + 2
            for col in range(2, len(month_headers) + 1):
                ws3.cell(row=row_num, column=col).number_format = money_fmt
            if i % 2 == 0:
                for col in range(1, len(month_headers) + 1):
                    ws3.cell(row=row_num, column=col).fill = alt_fill
            for col in range(1, len(month_headers) + 1):
                ws3.cell(row=row_num, column=col).border = thin_border

        auto_width(ws3)

        # ── Save ──
        wb.save(output_path)
        self._log("REPORT", f"Saved spreadsheet: {output_path}",
                   f"{len(receipts)} receipts, {len(cat_stats)} categories, "
                   f"{len(monthly)} months")
        return output_path

    def _generate_csv_fallback(self, receipts: List[ReceiptData],
                                output_path: str = None) -> str:
        """CSV fallback if openpyxl is not available."""
        import csv
        if output_path is None:
            if receipts and receipts[0].original_path:
                parent = Path(receipts[0].original_path).parent
            else:
                parent = self.base_path
            output_path = str(parent / "receipt_summary.csv")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Vendor", "Category", "Amount", "Tax",
                             "Currency", "Payment", "File"])
            for r in sorted(receipts, key=lambda x: x.date or "9999"):
                writer.writerow([r.date, r.vendor, r.category, r.amount,
                                 r.tax, r.currency, r.payment_method, r.filename])

        self._log("REPORT", f"Saved CSV fallback: {output_path}")
        return output_path

    # ── Phase 6: VERIFY ─────────────────────────────────────────────────────

    def verify_sample(self, receipts: List[ReceiptData],
                      sample_pct: float = 0.1) -> dict:
        """Phase 6: Spot-check OCR accuracy on a random sample."""
        self.progress.phase = "VERIFY"
        self.progress.phase_num = 6

        sample_size = max(1, int(len(receipts) * sample_pct))
        sample = random.sample(receipts, min(sample_size, len(receipts)))

        checks = []
        for r in sample:
            # Use organized_path if file was moved, else original_path
            check_path = r.organized_path if r.organized_path and Path(r.organized_path).exists() else r.original_path
            if not check_path or not Path(check_path).exists():
                continue

            # Re-read with verifier model
            verify_prompt = (
                f"Read this receipt image. What is the vendor name and total amount? "
                f"Reply as JSON: {{\"vendor\": \"...\", \"amount\": 0.00}}"
            )
            raw = _call_ollama_vision(check_path, verify_prompt,
                                       model=OCR_MODEL, timeout=30)
            vdata = _extract_json(raw)

            v_vendor = str(vdata.get("vendor", "")).lower().strip()
            v_amount = self._parse_amount(vdata.get("amount", 0))

            # Vendor match: substring or prefix match
            ocr_vendor = r.vendor.lower().strip()
            vendor_match = (
                v_vendor in ocr_vendor or ocr_vendor in v_vendor or
                (len(v_vendor) > 3 and len(ocr_vendor) > 3 and
                 v_vendor[:4] == ocr_vendor[:4])
            )
            # Amount match: within 50% tolerance (OCR re-reads vary)
            if r.amount > 0 and v_amount > 0:
                amount_diff_pct = abs(r.amount - v_amount) / max(r.amount, v_amount)
                amount_match = amount_diff_pct < 0.50
            elif r.amount == 0 and v_amount == 0:
                amount_match = True
            else:
                amount_match = False

            # Pass if EITHER vendor or amount matches (not both required)
            passed = vendor_match or amount_match

            checks.append({
                "file": r.filename,
                "ocr_vendor": r.vendor,
                "verify_vendor": vdata.get("vendor", ""),
                "vendor_match": vendor_match,
                "ocr_amount": r.amount,
                "verify_amount": v_amount,
                "amount_match": amount_match,
                "pass": passed,
            })

        pass_count = sum(1 for c in checks if c["pass"])
        total_amount = sum(r.amount for r in receipts)
        categories_found = list(set(r.category for r in receipts))

        result = {
            "total_processed": len(receipts),
            "sample_size": len(checks),
            "passed": pass_count,
            "failed": len(checks) - pass_count,
            "accuracy": round(pass_count / max(len(checks), 1) * 100, 1),
            "total_amount": round(total_amount, 2),
            "categories_found": categories_found,
            "category_count": len(categories_found),
            "flagged_items": [c for c in checks if not c["pass"]],
            "details": checks,
        }

        self._log("VERIFY", f"Spot-check complete: {pass_count}/{len(checks)} passed",
                   f"accuracy={result['accuracy']}%")
        return result

    # ── Full Pipeline ───────────────────────────────────────────────────────

    def process(self, folder_path: str, config: dict = None) -> dict:
        """
        Run the full 6-phase pipeline.

        config options:
            org_method: "move" | "copy" | "report_only"
            category_logic: "auto_detect" | "standard_business" | "personal_budget"
            output_dir: custom output directory
            verify: bool (default True)
        """
        config = config or {}
        org_method = config.get("org_method", "move")
        verify = config.get("verify", True)
        output_dir = config.get("output_dir", None)
        t0 = time.time()

        self.progress = ProcessingProgress(start_time=t0)

        # Phase 1: Scan
        scan_result = self.scan_folder(folder_path)
        if scan_result.get("error"):
            return {"error": scan_result["error"], "phase": "SCAN"}

        image_paths = scan_result["images"]
        if not image_paths:
            return {"error": "No receipt images found", "phase": "SCAN"}

        # Phase 2: OCR
        self.receipts = self.ocr_batch(image_paths)

        # Phase 3: Classify
        self.receipts = self.classify_receipts(self.receipts)

        # Phase 4: Organize (unless report_only)
        org_stats = {}
        if org_method != "report_only":
            org_stats = self.organize_files(self.receipts, output_dir, method=org_method)

        # Phase 5: Report
        report_path = self.generate_report(self.receipts)

        # Phase 6: Verify
        verify_result = {}
        if verify:
            verify_result = self.verify_sample(self.receipts)

        # Phase 7: Generate Analysis Report (.md)
        analysis_report_path = ""
        try:
            analysis_report_path = self._generate_analysis_report(
                self.receipts, scan_result, org_stats, verify_result,
                report_path, time.time() - t0, org_method
            )
        except Exception as e:
            self._log("REPORT", f"Analysis report generation failed: {e}")
            analysis_report_path = f"Error: {e}"

        # Final summary
        duration = time.time() - t0
        total_amount = sum(r.amount for r in self.receipts)
        categories = list(set(r.category for r in self.receipts))

        summary = {
            "success": True,
            "total_receipts": len(self.receipts),
            "total_amount": round(total_amount, 2),
            "categories": categories,
            "category_count": len(categories),
            "report_path": report_path,
            "analysis_report_path": analysis_report_path,
            "organization": org_stats,
            "verification": verify_result,
            "duration_sec": round(duration, 1),
            "audit_log_entries": len(self.audit_log),
            "avg_confidence": round(
                sum(r.ocr_confidence for r in self.receipts) / max(len(self.receipts), 1), 2
            ),
        }

        self._log("COMPLETE", f"Pipeline finished in {duration:.1f}s",
                   f"{len(self.receipts)} receipts, ${total_amount:.2f} total, "
                   f"{len(categories)} categories")

        return summary

    def _generate_analysis_report(self, receipts: List[ReceiptData],
                                   scan_result: dict, org_stats: dict,
                                   verify_result: dict, excel_path: str,
                                   duration: float, org_method: str) -> str:
        """Generate a comprehensive Markdown analysis report of the entire pipeline."""
        from datetime import datetime as dt

        # Determine output path
        if receipts and receipts[0].original_path:
            parent = Path(receipts[0].original_path).parent
        else:
            parent = self.base_path
        report_path = str(parent / "receipt_analysis_report.md")

        total_amount = sum(r.amount for r in receipts)
        avg_conf = sum(r.ocr_confidence for r in receipts) / max(len(receipts), 1)

        # Category stats
        cat_stats = {}
        for r in receipts:
            if r.category not in cat_stats:
                cat_stats[r.category] = {"count": 0, "total": 0.0, "receipts": []}
            cat_stats[r.category]["count"] += 1
            cat_stats[r.category]["total"] += r.amount
            cat_stats[r.category]["receipts"].append(r)

        lines = []
        lines.append("# Receipt Processing — Analysis Report")
        lines.append("")
        lines.append(f"**Generated**: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Pipeline Duration**: {duration:.1f} seconds")
        lines.append(f"**OCR Model**: {OCR_MODEL}")
        lines.append(f"**Classifier Model**: {CLASSIFIER_MODEL}")
        lines.append(f"**Organization Method**: {org_method}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ── Executive Summary ──
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Receipts | {len(receipts)} |")
        lines.append(f"| Total Amount | ${total_amount:,.2f} |")
        lines.append(f"| Categories | {len(cat_stats)} |")
        lines.append(f"| Avg OCR Confidence | {avg_conf:.0%} |")
        lines.append(f"| Files Organized | {org_stats.get('moved', 0)} |")
        lines.append(f"| Organization Errors | {org_stats.get('errors', 0)} |")
        if verify_result:
            lines.append(f"| Verification Accuracy | {verify_result.get('accuracy', 0)}% |")
        lines.append(f"| Excel Report | {Path(excel_path).name} |")
        lines.append("")

        # ── Phase 1: Scan Results ──
        lines.append("## Phase 1: Scan Results")
        lines.append("")
        lines.append(f"- **Folder**: `{scan_result.get('folder', 'N/A')}`")
        lines.append(f"- **Images Found**: {scan_result.get('image_count', scan_result.get('count', 0))}")
        lines.append(f"- **Batches**: {scan_result.get('batch_count', scan_result.get('batches', 0))}")
        lines.append(f"- **Total Size**: {scan_result.get('total_size_mb', 0)} MB")
        lines.append("")

        # ── Phase 2: OCR Results ──
        lines.append("## Phase 2: OCR Extraction Results")
        lines.append("")
        lines.append("| # | File | Vendor | Amount | Date | Confidence |")
        lines.append("|---|------|--------|--------|------|------------|")
        for i, r in enumerate(sorted(receipts, key=lambda x: x.date or "9999"), 1):
            vendor_short = (r.vendor[:25] + "...") if len(r.vendor) > 25 else r.vendor
            conf_emoji = "🟢" if r.ocr_confidence >= 0.7 else "🟡" if r.ocr_confidence >= 0.35 else "🔴"
            lines.append(
                f"| {i} | {r.filename} | {vendor_short} | "
                f"${r.amount:.2f} | {r.date or 'N/A'} | "
                f"{conf_emoji} {r.ocr_confidence:.0%} |"
            )
        lines.append("")

        # OCR Statistics
        high_conf = sum(1 for r in receipts if r.ocr_confidence >= 0.7)
        med_conf = sum(1 for r in receipts if 0.35 <= r.ocr_confidence < 0.7)
        low_conf = sum(1 for r in receipts if r.ocr_confidence < 0.35)
        with_vendor = sum(1 for r in receipts if r.vendor)
        with_amount = sum(1 for r in receipts if r.amount > 0)
        with_date = sum(1 for r in receipts if r.date)

        lines.append("### OCR Statistics")
        lines.append("")
        lines.append(f"| Metric | Count | % |")
        lines.append(f"|--------|-------|---|")
        lines.append(f"| High Confidence (>=70%) | {high_conf} | {high_conf/max(len(receipts),1):.0%} |")
        lines.append(f"| Medium Confidence (35-69%) | {med_conf} | {med_conf/max(len(receipts),1):.0%} |")
        lines.append(f"| Low Confidence (<35%) | {low_conf} | {low_conf/max(len(receipts),1):.0%} |")
        lines.append(f"| Vendor Extracted | {with_vendor} | {with_vendor/max(len(receipts),1):.0%} |")
        lines.append(f"| Amount Extracted | {with_amount} | {with_amount/max(len(receipts),1):.0%} |")
        lines.append(f"| Date Extracted | {with_date} | {with_date/max(len(receipts),1):.0%} |")
        lines.append("")

        # ── Phase 3: Classification Breakdown ──
        lines.append("## Phase 3: Classification Breakdown")
        lines.append("")
        lines.append("| Category | Count | Total Amount | Avg Amount | % of Spend |")
        lines.append("|----------|-------|-------------|------------|------------|")
        for cat, st in sorted(cat_stats.items(), key=lambda x: -x[1]["total"]):
            avg = st["total"] / max(st["count"], 1)
            pct = st["total"] / max(total_amount, 0.01) * 100
            lines.append(f"| {cat} | {st['count']} | ${st['total']:,.2f} | ${avg:,.2f} | {pct:.1f}% |")
        lines.append(f"| **TOTAL** | **{len(receipts)}** | **${total_amount:,.2f}** | "
                      f"**${total_amount/max(len(receipts),1):,.2f}** | **100%** |")
        lines.append("")

        # ── Phase 4: Organization ──
        lines.append("## Phase 4: File Organization")
        lines.append("")
        lines.append(f"- **Method**: `{org_method}`")
        lines.append(f"- **Files Moved**: {org_stats.get('moved', 0)}")
        lines.append(f"- **Errors**: {org_stats.get('errors', 0)}")
        lines.append(f"- **Folders Created**: {', '.join(org_stats.get('folders_created', []))}")
        lines.append("")

        if org_stats.get("folders_created"):
            lines.append("### Folder Structure")
            lines.append("```")
            lines.append("receipts/")
            for cat in sorted(org_stats.get("folders_created", [])):
                cat_count = cat_stats.get(cat, {}).get("count", 0)
                lines.append(f"  {cat}/  ({cat_count} files)")
            lines.append("  receipt_summary.xlsx")
            lines.append("  receipt_analysis_report.md")
            lines.append("```")
            lines.append("")

        # ── Phase 5: Excel Report ──
        lines.append("## Phase 5: Excel Report")
        lines.append("")
        lines.append(f"- **File**: `{Path(excel_path).name}`")
        lines.append("- **Sheets**:")
        lines.append("  1. **All_Receipts** — 15 columns: #, Date, Vendor, Category, Amount, Tax, Subtotal, Currency, Payment, Items, Receipt#, Confidence, Original File, Organized Path, Raw OCR")
        lines.append("  2. **Category_Summary** — Count, Total, Average, % per category")
        lines.append("  3. **Monthly_Breakdown** — Pivot by month x category")
        lines.append("")

        # ── Phase 6: Verification ──
        if verify_result:
            lines.append("## Phase 6: Verification Results")
            lines.append("")
            lines.append(f"- **Sample Size**: {verify_result.get('sample_size', 0)} receipts ({verify_result.get('sample_size', 0)/max(len(receipts),1):.0%} of total)")
            lines.append(f"- **Passed**: {verify_result.get('passed', 0)}")
            lines.append(f"- **Accuracy**: **{verify_result.get('accuracy', 0)}%**")
            lines.append("")

            flagged = verify_result.get("flagged_items", [])
            if flagged:
                lines.append("### Flagged Items")
                lines.append("")
                for item in flagged:
                    lines.append(f"- **{item.get('file', 'Unknown')}**: {item.get('reason', 'No reason')}")
                lines.append("")

        # ── Audit Trail ──
        lines.append("## Audit Trail")
        lines.append("")
        lines.append(f"Total log entries: {len(self.audit_log)}")
        lines.append("")
        lines.append("| Timestamp | Phase | Action | Details |")
        lines.append("|-----------|-------|--------|---------|")
        for entry in self.audit_log[-30:]:  # Last 30 entries
            ts = entry.get("timestamp", "")[:19]
            phase = entry.get("phase", "")
            action = entry.get("action", "")[:40]
            detail = entry.get("detail", "")[:60]
            lines.append(f"| {ts} | {phase} | {action} | {detail} |")
        if len(self.audit_log) > 30:
            lines.append(f"| ... | ... | *{len(self.audit_log) - 30} more entries* | ... |")
        lines.append("")

        lines.append("---")
        lines.append(f"*Report generated by Receipt Processor Pipeline v2.0*")

        # Write report
        report_content = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        self._log("REPORT", f"Saved analysis report: {report_path}")
        return report_path

    # ── Utilities ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_amount(val) -> float:
        """Parse a monetary amount from various formats."""
        if isinstance(val, (int, float)):
            return float(val)
        try:
            cleaned = re.sub(r'[^\d.]', '', str(val))
            return float(cleaned) if cleaned else 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _normalize_date(raw: str) -> str:
        """Normalize date to YYYY-MM-DD format."""
        if not raw:
            return ""
        raw = raw.strip()
        # Already in YYYY-MM-DD
        if re.match(r'^\d{4}-\d{2}-\d{2}$', raw):
            return raw
        # MM/DD/YYYY (4-digit year)
        m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', raw)
        if m:
            return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
        # MM/DD/YY (2-digit year) — the bug that caused path separator issue
        m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2})$', raw)
        if m:
            year = int(m.group(3))
            year = year + 2000 if year < 70 else year + 1900  # 25→2025, 99→1999
            return f"{year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
        # DD-MM-YYYY or DD/MM/YYYY (European, 4-digit year)
        m = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$', raw)
        if m:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if month > 12:
                month, day = day, month
            return f"{year}-{month:02d}-{day:02d}"
        # DD-MM-YY (2-digit year)
        m = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{2})$', raw)
        if m:
            day, month = int(m.group(1)), int(m.group(2))
            year = int(m.group(3))
            year = year + 2000 if year < 70 else year + 1900
            if month > 12:
                month, day = day, month
            return f"{year}-{month:02d}-{day:02d}"
        # Try parsing with datetime
        for fmt in ["%B %d, %Y", "%b %d, %Y", "%m-%d-%Y", "%Y/%m/%d",
                    "%m-%d-%y", "%b %d, %y"]:
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Last resort: replace any slashes with dashes to prevent path issues
        return raw.replace('/', '-')

    def get_progress(self) -> dict:
        """Return current processing progress."""
        return {
            "phase": self.progress.phase,
            "phase_num": self.progress.phase_num,
            "total_phases": self.progress.total_phases,
            "batch": f"{self.progress.batch_current}/{self.progress.batch_total}",
            "files": f"{self.progress.files_processed}/{self.progress.files_total}",
            "current_file": self.progress.current_file,
            "status": self.progress.status_line(),
            "errors": self.progress.errors,
        }

    def get_receipts(self) -> List[dict]:
        """Return all processed receipts as dicts."""
        return [r.to_dict() for r in self.receipts]
