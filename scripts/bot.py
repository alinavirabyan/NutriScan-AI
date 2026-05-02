import os
import io
import sys
import json
import logging
import tempfile
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardMarkup, BotCommand, ReplyKeyboardRemove,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, filters,
    ContextTypes,
)
from telegram.constants import ChatAction, ParseMode

import cv2
import numpy as np
from PIL import Image
import pytesseract

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
TESS_DIR     = os.getenv("TESSERACT_MODEL_DIR", "")
if TESS_DIR:
    os.environ["TESSDATA_PREFIX"] = TESS_DIR

PRIMARY_LANG  = "eng_textocr"
FALLBACK_LANG = "eng"

# ── Conversation states ────────────────────────────────────────
(NAME, AGE, WEIGHT, ALLERGENS,
 EDIT_FIELD, EDIT_NAME, EDIT_AGE, EDIT_WEIGHT, EDIT_ALLERGENS,
 ASK_AI, ADD_CUSTOM_ALLERGEN, EDIT_ADD_CUSTOM_ALLERGEN) = range(12)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════

DB_FILE = Path(__file__).parent / "user_data.json"

def load_db():
    if DB_FILE.exists():
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {"users": {}, "scans": {}}

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def get_user(uid: int):
    return load_db()["users"].get(str(uid), {})

def save_user(uid: int, data: dict):
    db = load_db()
    db["users"][str(uid)] = data
    save_db(db)

def save_scan(uid: int, scan_data: dict):
    db = load_db()
    db["scans"].setdefault(str(uid), []).append(scan_data)
    save_db(db)

def get_scans(uid: int):
    return load_db()["scans"].get(str(uid), [])

def delete_scan(uid: int, index: int):
    """Delete a scan by index from user's history."""
    db = load_db()
    scans = db["scans"].get(str(uid), [])
    if 0 <= index < len(scans):
        scans.pop(index)
        db["scans"][str(uid)] = scans
        save_db(db)
        return True
    return False

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

RATING_EMOJI = {"healthy": "🟢", "moderate": "🟡", "unhealthy": "🔴", "unknown": "⚪"}
RATING_LABEL = {"healthy": "Healthy", "moderate": "Moderate", "unhealthy": "Unhealthy", "unknown": "Unknown"}

RATING_VERDICT = {
    "healthy":   "✅ Good news! This product has a <b>healthy nutritional profile</b>. It's a solid choice for regular consumption.",
    "moderate":  "⚠️ This product is <b>okay in moderation</b>. It has some concerns — check the warnings below before making it a daily habit.",
    "unhealthy": "🚫 <b>Not recommended for regular use.</b> This product contains ingredients or nutrients that may negatively affect your health over time.",
    "unknown":   "🔍 Health rating could not be fully determined. Review the ingredients and warnings carefully.",
}

# Standard built-in allergens
ALLERGEN_LIST = ["milk", "gluten", "peanut", "soy", "egg", "lactose", "nuts", "wheat"]
ALLERGEN_EMOJIS = {
    "milk": "🥛", "gluten": "🌾", "peanut": "🥜", "soy": "🫘",
    "egg": "🥚", "lactose": "🥛", "nuts": "🥜", "wheat": "🌾",
}

ALLERGEN_INFO = {
    "milk":    "🥛 <b>Milk Allergy</b>\n\n━━━━━━━━━━━━━━━━━━━━\nAn immune reaction to proteins in cow's milk.\n\n😰 <b>Symptoms:</b> Hives, wheezing, vomiting, and in severe cases anaphylaxis.\n\n🛒 <b>Commonly found in:</b>\n<i>Dairy products, cheese, butter, cream, yogurt, whey.</i>",
    "gluten":  "🌾 <b>Gluten Sensitivity</b>\n\n━━━━━━━━━━━━━━━━━━━━\nCeliac disease affects ~1% of people worldwide.\n\n😰 <b>Symptoms:</b> Bloating, diarrhea, fatigue, long-term nutrient deficiency.\n\n🛒 <b>Commonly found in:</b>\n<i>Bread, pasta, cereals, beer, sauces.</i>",
    "peanut":  "🥜 <b>Peanut Allergy</b>\n\n━━━━━━━━━━━━━━━━━━━━\nOne of the most severe food allergies.\n\n⚠️ <b>Warning:</b> Even trace amounts can trigger life-threatening anaphylaxis. Often lifelong.\n\n🛒 <b>Commonly found in:</b>\n<i>Peanut butter, snacks, Asian cuisine, baked goods.</i>",
    "soy":     "🫘 <b>Soy Allergy</b>\n\n━━━━━━━━━━━━━━━━━━━━\nCommon in infants and young children.\n\n😰 <b>Symptoms:</b> Rash, itching, digestive upset. Most children outgrow it by age 3–5.\n\n🛒 <b>Commonly found in:</b>\n<i>Soy sauce, tofu, edamame, processed foods.</i>",
    "egg":     "🥚 <b>Egg Allergy</b>\n\n━━━━━━━━━━━━━━━━━━━━\nSecond most common food allergy in children.\n\n😰 <b>Symptoms:</b> Mild rash to severe anaphylaxis. Many children outgrow it.\n\n🛒 <b>Commonly found in:</b>\n<i>Baked goods, mayonnaise, pasta, sauces.</i>",
    "lactose": "🥛 <b>Lactose Intolerance</b>\n\n━━━━━━━━━━━━━━━━━━━━\nAffects up to 65% of the world population.\n\n😰 <b>Symptoms:</b> Bloating, gas, diarrhea. Not an allergy — a digestive intolerance.\n\n🛒 <b>Commonly found in:</b>\n<i>Milk, cheese, ice cream, cream-based products.</i>",
    "nuts":    "🥜 <b>Tree Nut Allergy</b>\n\n━━━━━━━━━━━━━━━━━━━━\nIncludes almonds, walnuts, cashews, pistachios.\n\n⚠️ <b>Warning:</b> Often lifelong. Can cause severe anaphylaxis. Distinct from peanut allergy.\n\n🛒 <b>Commonly found in:</b>\n<i>Nut butters, chocolates, baked goods, trail mixes.</i>",
    "wheat":   "🌾 <b>Wheat Allergy</b>\n\n━━━━━━━━━━━━━━━━━━━━\nAn immune reaction to wheat proteins — different from celiac disease.\n\n😰 <b>Symptoms:</b> Swelling, itching, breathing difficulties, digestive issues.\n\n🛒 <b>Commonly found in:</b>\n<i>Bread, pasta, cereals, soy sauce, couscous.</i>",
}

RATING_COLOR = {
    "healthy":   "#2e7d32",
    "moderate":  "#e65100",
    "unhealthy": "#b71c1c",
    "unknown":   "#546e7a",
}

# ═══════════════════════════════════════════════════════════════
# OCR ENGINE
# ═══════════════════════════════════════════════════════════════

_easyocr_reader = None

def _get_easy_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader

def enhance_for_ocr(image_path: str) -> Image.Image:
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        img_pil = Image.open(image_path).convert("RGB")
        img_cv  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    if w < 1200:
        scale  = 1200 / w
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray   = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(binary)

def run_ocr(image_path: str) -> dict:
    enhanced = enhance_for_ocr(image_path)
    tess = ""
    for lang in (PRIMARY_LANG, FALLBACK_LANG):
        for psm in (6, 7, 11, 3):
            try:
                t = pytesseract.image_to_string(enhanced, lang=lang, config=f"--psm {psm}").strip()
                log.info(f"Tesseract lang={lang} psm={psm} → {len(t)} chars")
                if len(t) > len(tess):
                    tess = t
            except Exception as e:
                log.warning(f"Tesseract lang={lang} psm={psm} failed: {e}")
    easy = ""
    try:
        reader  = _get_easy_reader()
        img_arr = np.array(Image.open(image_path).convert("RGB"))
        easy    = " ".join(txt for _, txt, _ in reader.readtext(img_arr)).strip()
        log.info(f"EasyOCR → {len(easy)} chars")
    except Exception as e:
        log.warning(f"EasyOCR failed: {e}")

    if tess and len(tess) >= len(easy or ""):
        text, engine = tess, "Tesseract"
    elif easy:
        text, engine = easy, "EasyOCR"
    elif tess:
        text, engine = tess, "Tesseract"
    else:
        text, engine = "", "none"

    log.info(f"OCR: engine={engine}, chars={len(text)}")
    return {"text": text, "engine": engine, "tesseract": tess, "easyocr": easy}

# ═══════════════════════════════════════════════════════════════
# OLLAMA LLM
# ═══════════════════════════════════════════════════════════════

def ollama_alive() -> bool:
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
    except Exception:
        return False

def call_llm(prompt: str, timeout: int = 120) -> str:
    try:
        r = requests.post(OLLAMA_URL, json={
            "model":    OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
        }, timeout=timeout)
        return r.json()["message"]["content"].strip() if r.status_code == 200 else f"Error {r.status_code}"
    except Exception as e:
        return f"❌ LLM error: {e}"

def analyze_label(ocr_text: str, user_profile: dict = None) -> dict:
    if len(ocr_text.strip()) < 10:
        return {
            "is_food_label": False, "reason": "Not enough text was extracted.",
            "health_rating": "unknown", "summary": "", "ingredients": [],
            "allergens": [], "nutrition": "", "warnings": [], "positives": [],
        }
    user_context = ""
    if user_profile:
        user_context = (
            f"\nUser Profile: {user_profile.get('name','')}, "
            f"{user_profile.get('age','')} years old, "
            f"{user_profile.get('weight','')}kg\n"
            f"Allergies: {', '.join(user_profile.get('allergens',[])) or 'none'}"
        )
    prompt = f"""You are a certified nutrition and food safety expert. Analyze this food label OCR text.{user_context}

OCR TEXT:
{ocr_text[:2500]}

Reply ONLY using this exact format:
IS_FOOD_LABEL: yes/no
REASON: <one sentence>
HEALTH_RATING: healthy/moderate/unhealthy
SUMMARY: <2-3 sentences explaining WHY this rating was given, mentioning specific nutrients>
INGREDIENTS: <comma-separated list>
ALLERGENS: <comma-separated or "none">
NUTRITION: <one line summary of key nutritional values>
WARNINGS: <comma-separated health concerns or "none">
POSITIVES: <comma-separated benefits or "none">"""

    raw    = call_llm(prompt)
    result = {
        "is_food_label": False, "reason": "Analysis failed.", "health_rating": "unknown",
        "summary": "", "ingredients": [], "allergens": [], "nutrition": "",
        "warnings": [], "positives": [], "raw": raw,
    }
    for line in raw.splitlines():
        line = line.strip()
        def v(p): return line.replace(p, "").strip()
        if   line.startswith("IS_FOOD_LABEL:"):  result["is_food_label"] = v("IS_FOOD_LABEL:").lower() in ("yes","true","1")
        elif line.startswith("REASON:"):          result["reason"]      = v("REASON:")
        elif line.startswith("HEALTH_RATING:"):
            r = v("HEALTH_RATING:").lower()
            if r in ("healthy","moderate","unhealthy"): result["health_rating"] = r
        elif line.startswith("SUMMARY:"):         result["summary"]     = v("SUMMARY:")
        elif line.startswith("INGREDIENTS:"):     result["ingredients"] = [i.strip() for i in v("INGREDIENTS:").split(",") if i.strip()]
        elif line.startswith("ALLERGENS:"):       result["allergens"]   = [a.strip() for a in v("ALLERGENS:").split(",") if a.strip() and a.lower() != "none"]
        elif line.startswith("NUTRITION:"):       result["nutrition"]   = v("NUTRITION:")
        elif line.startswith("WARNINGS:"):        result["warnings"]    = [w.strip() for w in v("WARNINGS:").split(",") if w.strip() and w.lower() != "none"]
        elif line.startswith("POSITIVES:"):       result["positives"]   = [p.strip() for p in v("POSITIVES:").split(",") if p.strip() and p.lower() != "none"]
    return result

def answer_question(question: str, analysis: dict, user_profile: dict = None) -> str:
    context = (
        f"Product Summary: {analysis.get('summary','Unknown')}\n"
        f"Ingredients: {', '.join(analysis.get('ingredients',[]))}\n"
        f"Allergens: {', '.join(analysis.get('allergens',[])) or 'none'}\n"
        f"Health Rating: {analysis.get('health_rating','unknown')}"
    )
    prompt = f"""You are a nutrition expert. A user has a question about a food product they scanned.

PRODUCT INFORMATION:
{context}

USER QUESTION: {question}

Give a clear, helpful answer in 2-4 sentences."""
    return call_llm(prompt, timeout=60)

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def get_all_allergens(user_profile: dict) -> list:
    """Return combined built-in + custom allergens for a user."""
    standard = user_profile.get("allergens", [])
    custom   = user_profile.get("custom_allergens", [])
    return standard + custom

def format_allergen_display(allergen: str) -> str:
    """Return emoji + name for any allergen (built-in or custom)."""
    emoji = ALLERGEN_EMOJIS.get(allergen.lower(), "⚠️")
    return f"{emoji} {allergen.capitalize()}"

# ═══════════════════════════════════════════════════════════════
# PDF GENERATOR
# ═══════════════════════════════════════════════════════════════

def build_pdf(ocr_text: str, analysis: dict, user_profile: dict = None) -> bytes:
    """
    Generate a styled PDF report.
    Sections: Header → Profile → Health Rating → Summary →
              Ingredients → Allergens → Nutrition → Warnings →
              Positives → OCR text (small, subtle, at the very end)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    def h1(t):
        return Paragraph(t, ParagraphStyle("h1", parent=styles["Title"], fontSize=22,
                                           textColor=colors.HexColor("#1a237e"), spaceAfter=4))
    def h2(t):
        return Paragraph(t, ParagraphStyle("h2", parent=styles["Heading2"], fontSize=14,
                                           textColor=colors.HexColor("#283593"), spaceBefore=10, spaceAfter=4))
    def body(t, c="#212121"):
        return Paragraph(t, ParagraphStyle("body", parent=styles["Normal"], fontSize=11,
                                           textColor=colors.HexColor(c), leading=16))
    def small(t, c="#9e9e9e"):
        """Small subtle text — used for OCR dump at the very end."""
        return Paragraph(t, ParagraphStyle("small", parent=styles["Normal"], fontSize=7,
                                           textColor=colors.HexColor(c), leading=10))
    def hr():
        return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c5cae9"))

    rating  = analysis.get("health_rating", "unknown")
    r_color = RATING_COLOR.get(rating, "#546e7a")

    # ── Header ─────────────────────────────────────────────────
    story += [
        h1("🍎 NutriScan AI — Food Label Report"),
        body(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "#757575"),
        hr(),
        Spacer(1, 0.3*cm),
    ]

    # ── User Profile ───────────────────────────────────────────
    if user_profile:
        all_allergens = get_all_allergens(user_profile)
        al_str = ", ".join(all_allergens) if all_allergens else "None"
        story += [
            h2("👤 User Profile"),
            body(f"Name: {user_profile.get('name','N/A')}  |  "
                 f"Age: {user_profile.get('age','N/A')}  |  "
                 f"Weight: {user_profile.get('weight','N/A')} kg"),
            body(f"Personal Allergens: {al_str}"),
            Spacer(1, 0.2*cm),
        ]

    # ── Health Rating ──────────────────────────────────────────
    story += [
        h2("📊 Health Rating"),
        Paragraph(
            f'<font color="{r_color}"><b>{rating.capitalize()}</b></font>',
            ParagraphStyle("rating", parent=styles["Normal"], fontSize=16, leading=22)
        ),
        Spacer(1, 0.2*cm),
    ]

    # ── Summary (WHY this rating) ──────────────────────────────
    if analysis.get("summary"):
        story += [
            h2("📝 Summary"),
            body(analysis["summary"]),
            Spacer(1, 0.2*cm),
        ]

    # ── Ingredients ────────────────────────────────────────────
    if analysis.get("ingredients"):
        story.append(h2("🧪 Ingredients"))
        for ing in analysis["ingredients"]:
            story.append(body(f"  • {ing}"))
        story.append(Spacer(1, 0.2*cm))

    # ── Allergens ──────────────────────────────────────────────
    story.append(h2("⚠️ Allergens Detected"))
    if analysis.get("allergens"):
        for a in analysis["allergens"]:
            story.append(body(f"  ⚠️ {a}", "#b71c1c"))
    else:
        story.append(body("  ✓ No common allergens detected", "#2e7d32"))
    story.append(Spacer(1, 0.2*cm))

    # ── Nutrition ──────────────────────────────────────────────
    if analysis.get("nutrition"):
        story += [h2("🥗 Nutrition Facts"), body(analysis["nutrition"]), Spacer(1, 0.2*cm)]

    # ── Warnings ───────────────────────────────────────────────
    if analysis.get("warnings"):
        story.append(h2("🚫 Health Warnings"))
        for w in analysis["warnings"]:
            story.append(body(f"  ✗ {w}", "#b71c1c"))
        story.append(Spacer(1, 0.2*cm))

    # ── Positives ──────────────────────────────────────────────
    if analysis.get("positives"):
        story.append(h2("✅ Positive Aspects"))
        for p in analysis["positives"]:
            story.append(body(f"  ✓ {p}", "#2e7d32"))
        story.append(Spacer(1, 0.2*cm))

    # ── OCR Text — small subtle text at the very end ───────────
    story += [
        hr(),
        Spacer(1, 0.5*cm),
        Paragraph(
            "Raw OCR Extracted Text",
            ParagraphStyle("ocr_title", parent=styles["Normal"], fontSize=8,
                           textColor=colors.HexColor("#bdbdbd"), leading=12)
        ),
        Spacer(1, 0.1*cm),
        small(ocr_text[:3000].replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")),
        Spacer(1, 0.3*cm),
        hr(),
        small("Generated by NutriScan AI  •  Powered by Fine-tuned Tesseract + Ollama  •  " +
              datetime.now().strftime("%Y-%m-%d")),
    ]

    doc.build(story)
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════
# KEYBOARDS
# ═══════════════════════════════════════════════════════════════

def main_menu_kb():
    return ReplyKeyboardMarkup([
        ["📸 Scan a Label",  "👤 My Profile"],
        ["📊 History",       "⚙️ Allergens"],
        ["❓ Help"],
    ], resize_keyboard=True)

def setup_menu_kb():
    return ReplyKeyboardMarkup([["🚀 Set Up Profile & Start"]], resize_keyboard=True)

def age_kb(prefix="age"):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧒 Under 18", callback_data=f"{prefix}_under18"),
         InlineKeyboardButton("🧑 18–25",    callback_data=f"{prefix}_18_25")],
        [InlineKeyboardButton("🧑 26–35",    callback_data=f"{prefix}_26_35"),
         InlineKeyboardButton("🧑 36–50",    callback_data=f"{prefix}_36_50")],
        [InlineKeyboardButton("🧓 50+",      callback_data=f"{prefix}_50plus")],
    ])

def allergen_kb(selected: list = None, custom: list = None,
                prefix: str = "allergen", show_info: bool = False):
    """
    Build the allergen selection keyboard.
    Shows all standard allergens + any custom ones the user added.
    At the bottom: ➕ Add my own allergen + ✅ Done.
    """
    if selected is None: selected = []
    if custom   is None: custom   = []

    buttons = []

    # Standard allergens
    for a in ALLERGEN_LIST:
        emoji = ALLERGEN_EMOJIS.get(a, "⚠️")
        mark  = "✅" if a in selected else "⬜"
        cb    = f"{prefix}_info_{a}" if show_info else f"{prefix}_{a}"
        buttons.append([InlineKeyboardButton(f"{mark} {emoji} {a.capitalize()}", callback_data=cb)])

    # Custom allergens (user-added) — always toggle directly, no info page
    for ca in custom:
        mark = "✅" if ca in selected else "⬜"
        buttons.append([InlineKeyboardButton(f"{mark} ⚠️ {ca.capitalize()} (custom)",
                                             callback_data=f"{prefix}_toggle_{ca}")])

    # Add custom + Done
    buttons.append([InlineKeyboardButton("➕ Add my own allergen", callback_data=f"{prefix}_add_custom")])
    buttons.append([InlineKeyboardButton("✅ Done — Save My Allergens", callback_data=f"{prefix}_done")])
    return InlineKeyboardMarkup(buttons)

def allergen_info_view_kb(allergen: str, prefix: str = "allergen",
                          selected: list = None, in_edit: bool = False):
    if selected is None: selected = []
    already      = allergen in selected
    toggle_label = "✅ Remove from my list" if already else "➕ Add to my list"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(toggle_label,      callback_data=f"{prefix}_toggle_{allergen}")],
        [InlineKeyboardButton("🔙 Back to list", callback_data=f"{prefix}_back_list")],
    ])

def allergen_main_menu_kb():
    buttons = []
    for a in ALLERGEN_LIST:
        emoji = ALLERGEN_EMOJIS.get(a, "⚠️")
        buttons.append([InlineKeyboardButton(f"{emoji} {a.capitalize()} — tap to learn more",
                                             callback_data=f"ainfo_{a}")])
    return InlineKeyboardMarkup(buttons)

def after_scan_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 Download PDF",    callback_data="pdf"),
         InlineKeyboardButton("💬 Ask AI",          callback_data="ask_ai")],
        [InlineKeyboardButton("🔄 New Scan",        callback_data="new_scan"),
         InlineKeyboardButton("⭐ Save to History", callback_data="save")],
    ])

def history_list_kb(scans: list):
    """Build inline keyboard — one button per saved scan (date + product)."""
    buttons = []
    # Scans are stored oldest-first; show newest first
    reversed_scans = list(enumerate(scans))[::-1][:10]  # max 10
    for original_index, s in reversed_scans:
        date    = s.get("date", "")[:10]
        product = s.get("product", "Unknown product")[:35]
        rating  = s.get("rating", s.get("analysis", {}).get("health_rating", "unknown"))
        emoji   = RATING_EMOJI.get(rating, "⚪")
        label   = f"{emoji} {date} — {product}"
        buttons.append([InlineKeyboardButton(label, callback_data=f"hist_view_{original_index}")])
    return InlineKeyboardMarkup(buttons)

def scan_detail_kb(scan_index: int):
    """Buttons shown under a scan detail view."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗑️ Delete this scan", callback_data=f"hist_delete_{scan_index}")],
        [InlineKeyboardButton("🔙 Back to History",  callback_data="hist_back")],
    ])

def profile_edit_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✏️ Edit Name",      callback_data="edit_name")],
        [InlineKeyboardButton("🎂 Edit Age",       callback_data="edit_age")],
        [InlineKeyboardButton("⚖️ Edit Weight",    callback_data="edit_weight")],
        [InlineKeyboardButton("⚙️ Edit Allergens", callback_data="edit_allergens")],
        [InlineKeyboardButton("💾 Save & Close",   callback_data="edit_done")],
    ])

# ═══════════════════════════════════════════════════════════════
# SESSION STORE
# ═══════════════════════════════════════════════════════════════

sessions: Dict[int, dict] = {}

# ═══════════════════════════════════════════════════════════════
# HELPER — format scan detail message
# ═══════════════════════════════════════════════════════════════

def format_scan_detail(s: dict) -> str:
    """Format a saved scan dict into a rich message string."""
    analysis = s.get("analysis", {})
    rating   = s.get("rating", analysis.get("health_rating", "unknown"))
    emoji    = RATING_EMOJI.get(rating, "⚪")
    label    = RATING_LABEL.get(rating, "Unknown")
    date     = s.get("date", "")[:16].replace("T", " ")

    verdict = RATING_VERDICT.get(rating, "")

    # Summary
    summary = analysis.get("summary", "")
    summary_block = f"\n📝 <b>Summary</b>\n{summary}\n" if summary else ""

    # Ingredients
    ingredients = analysis.get("ingredients", [])
    if ingredients:
        ing_lines = "\n".join(f"  • {i}" for i in ingredients[:20])
        ing_block = f"\n🧪 <b>Ingredients</b>\n{ing_lines}\n"
    else:
        ing_block = ""

    # Allergens
    allergens = analysis.get("allergens", [])
    if allergens:
        al_lines  = "\n".join(f"  ⚠️ {a}" for a in allergens)
        al_block  = f"\n🚨 <b>Allergens Detected</b>\n{al_lines}\n"
    else:
        al_block  = "\n✅ <b>No allergens detected</b>\n"

    # Warnings
    warnings = analysis.get("warnings", [])
    if warnings:
        w_lines   = "\n".join(f"  🚫 {w}" for w in warnings)
        w_block   = f"\n⚠️ <b>Warnings</b>\n{w_lines}\n"
    else:
        w_block   = ""

    # Positives
    positives = analysis.get("positives", [])
    if positives:
        p_lines   = "\n".join(f"  ✅ {p}" for p in positives)
        p_block   = f"\n💚 <b>Positives</b>\n{p_lines}\n"
    else:
        p_block   = ""

    return (
        f"📅 <b>Scanned on:</b> {date}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} <b>Health Rating: {label}</b>\n\n"
        f"{verdict}\n"
        f"{summary_block}"
        f"{ing_block}"
        f"{al_block}"
        f"{w_block}"
        f"{p_block}"
        f"━━━━━━━━━━━━━━━━━━━━"
    )

# ═══════════════════════════════════════════════════════════════
# /start
# ═══════════════════════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user    = update.effective_user
    uid     = user.id
    profile = get_user(uid)

    if not profile:
        await update.message.reply_text(
            f"👋 <b>Hi {user.first_name}! I'm NutriScan AI</b> 🍎\n\n"
            f"I'm your personal food label assistant.\n"
            f"I can help you:\n\n"
            f"🔍 Understand what's <b>really</b> in your food\n"
            f"🔥 Track calories & nutrition facts\n"
            f"🚨 Detect <b>your personal allergens</b> instantly\n"
            f"🟢🟡🔴 Get a clear <b>health rating</b> for any product\n"
            f"📄 Download a full <b>PDF report</b>\n"
            f"💬 Ask me <b>any nutrition question</b> about what you scan\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Let's get started! First I'll set up your profile\n"
            f"so I can give you <b>personalised advice</b> 👇",
            parse_mode=ParseMode.HTML,
            reply_markup=setup_menu_kb()
        )
    else:
        await update.message.reply_text(
            f"👋 <b>Welcome back, {profile.get('name', user.first_name)}!</b> 🍎\n\n"
            f"Ready to scan another label?\n\n"
            f"📸 Just send me a food label photo\n"
            f"and I'll analyse it in seconds!",
            parse_mode=ParseMode.HTML,
            reply_markup=main_menu_kb()
        )
    return ConversationHandler.END

# ═══════════════════════════════════════════════════════════════
# PROFILE SETUP CONVERSATION
# ═══════════════════════════════════════════════════════════════

async def setup_intro(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🍎 <b>Let's set up your profile!</b>\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "I'll use your age, weight and allergen list\n"
        "to give you <b>personalised nutrition advice</b>\n"
        "every time you scan a product.\n\n"
        "👤 <b>What's your name?</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardRemove()
    )
    return NAME

async def get_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["name"] = update.message.text.strip()
    await update.message.reply_text(
        f"🎉 Nice to meet you, <b>{ctx.user_data['name']}</b>!\n\n"
        f"🎂 <b>How old are you?</b>\n\n"
        f"<i>This helps me personalise your nutrition advice</i>",
        parse_mode=ParseMode.HTML,
        reply_markup=age_kb()
    )
    return AGE

async def get_age(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    raw = query.data.replace("age_", "").replace("_", "–")
    ctx.user_data["age"] = raw
    await query.edit_message_text(
        f"✅ Age saved: <b>{raw}</b>\n\n"
        f"⚖️ <b>What's your weight in kg?</b>\n\n"
        f"<i>e.g. type</i> <code>70</code>",
        parse_mode=ParseMode.HTML
    )
    return WEIGHT

async def get_weight(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data["weight"] = int(update.message.text.strip())
    except ValueError:
        await update.message.reply_text(
            "⚠️ Please enter a number only — e.g. <code>70</code>",
            parse_mode=ParseMode.HTML
        )
        return WEIGHT

    ctx.user_data["allergens"]        = []
    ctx.user_data["custom_allergens"] = []
    await update.message.reply_text(
        f"✅ Weight saved: <b>{ctx.user_data['weight']} kg</b>\n\n"
        f"⚠️ <b>Do you have any food allergies?</b>\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"• Tap an allergen to <b>read about it</b>\n"
        f"• Then add it to your personal list\n"
        f"• Use <b>➕ Add my own</b> for unlisted allergens\n\n"
        f"<i>I'll warn you when I detect your allergens\nin any product you scan!</i>",
        parse_mode=ParseMode.HTML,
        reply_markup=allergen_kb([], [], prefix="allergen", show_info=True)
    )
    return ALLERGENS

async def allergen_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handles all allergen callbacks during profile setup."""
    query    = update.callback_query
    await query.answer()
    data     = query.data
    selected = ctx.user_data.get("allergens", [])
    custom   = ctx.user_data.get("custom_allergens", [])

    # ── Back to list — MUST be before startswith checks ───────
    if data == "allergen_back_list":
        count = len(selected) + len(custom)
        await query.edit_message_text(
            f"⚠️ <b>Food Allergies</b>\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{'✅ ' + str(count) + ' allergen(s) selected' if count else '⬜ None selected yet'}\n\n"
            f"Tap any allergen to read about it,\nthen add it to your list:",
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_kb(selected, custom, prefix="allergen", show_info=True)
        )
        return ALLERGENS

    # ── Add custom allergen ────────────────────────────────────
    if data == "allergen_add_custom":
        await query.edit_message_text(
            "➕ <b>Add Your Own Allergen</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Type the name of your allergen below.\n\n"
            "<i>Examples: sesame, shellfish, mustard,\nsulphites, celery, lupin...</i>",
            parse_mode=ParseMode.HTML
        )
        ctx.user_data["awaiting_custom_allergen_context"] = "setup"
        return ADD_CUSTOM_ALLERGEN

    # ── Show allergen info ─────────────────────────────────────
    if data.startswith("allergen_info_"):
        allergen = data.replace("allergen_info_", "")
        info     = ALLERGEN_INFO.get(allergen, "No information available.")
        await query.edit_message_text(
            info, parse_mode=ParseMode.HTML,
            reply_markup=allergen_info_view_kb(allergen, prefix="allergen", selected=selected)
        )
        return ALLERGENS

    # ── Toggle allergen ────────────────────────────────────────
    if data.startswith("allergen_toggle_"):
        allergen = data.replace("allergen_toggle_", "")
        # Could be standard or custom
        if allergen in ALLERGEN_LIST:
            if allergen in selected: selected.remove(allergen)
            else: selected.append(allergen)
            ctx.user_data["allergens"] = selected
            info = ALLERGEN_INFO.get(allergen, "")
            await query.edit_message_text(
                info, parse_mode=ParseMode.HTML,
                reply_markup=allergen_info_view_kb(allergen, prefix="allergen", selected=selected)
            )
        else:
            # Custom allergen toggle — go back to list
            all_selected = selected + [a for a in custom if a not in selected]
            if allergen in all_selected: all_selected.remove(allergen)
            else: all_selected.append(allergen)
            ctx.user_data["allergens"] = [a for a in all_selected if a in ALLERGEN_LIST]
            # For custom ones, we track which are "active" via selected list too
            if allergen in selected: selected.remove(allergen)
            else: selected.append(allergen)
            ctx.user_data["allergens"] = selected
            await query.edit_message_text(
                f"⚠️ <b>Food Allergies</b>\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Tap any allergen to read about it,\nthen add it to your list:",
                parse_mode=ParseMode.HTML,
                reply_markup=allergen_kb(selected, custom, prefix="allergen", show_info=True)
            )
        return ALLERGENS

    # ── Done ───────────────────────────────────────────────────
    if data == "allergen_done":
        uid     = query.from_user.id
        profile = {
            "name":             ctx.user_data.get("name", ""),
            "age":              ctx.user_data.get("age", ""),
            "weight":           ctx.user_data.get("weight", ""),
            "allergens":        selected,
            "custom_allergens": custom,
            "created":          datetime.now().isoformat(),
        }
        save_user(uid, profile)
        all_al = selected + custom
        al_display = " • ".join(format_allergen_display(a) for a in all_al) if all_al else "None"
        await query.edit_message_text(
            f"🎉 <b>Profile Created!</b>\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>Name:</b>      {profile['name']}\n"
            f"🎂 <b>Age:</b>       {profile['age']}\n"
            f"⚖️ <b>Weight:</b>    {profile['weight']} kg\n"
            f"⚠️ <b>Allergens:</b> {al_display}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🍎 All set! Send me a food label photo to begin 📸",
            parse_mode=ParseMode.HTML
        )
        await query.message.reply_text("👇 Use the menu below:", reply_markup=main_menu_kb())
        return ConversationHandler.END

    return ALLERGENS

async def receive_custom_allergen(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Receive a custom allergen typed by the user during setup."""
    allergen_name = update.message.text.strip().lower()
    custom        = ctx.user_data.get("custom_allergens", [])
    selected      = ctx.user_data.get("allergens", [])

    if allergen_name and allergen_name not in custom and allergen_name not in ALLERGEN_LIST:
        custom.append(allergen_name)
        ctx.user_data["custom_allergens"] = custom
        # Auto-select it
        selected.append(allergen_name)
        ctx.user_data["allergens"] = selected
        conf = f"✅ <b>'{allergen_name.capitalize()}'</b> added to your allergen list!"
    elif allergen_name in ALLERGEN_LIST:
        conf = f"ℹ️ <b>'{allergen_name.capitalize()}'</b> is already in the standard list — tap it to add it."
    elif allergen_name in custom:
        conf = f"ℹ️ <b>'{allergen_name.capitalize()}'</b> is already in your custom list."
    else:
        conf = "⚠️ Please enter a valid allergen name."

    await update.message.reply_text(
        f"{conf}\n\n"
        f"You can add more or tap ✅ Done when finished:",
        parse_mode=ParseMode.HTML,
        reply_markup=allergen_kb(selected, custom, prefix="allergen", show_info=True)
    )
    return ALLERGENS

# ═══════════════════════════════════════════════════════════════
# PROFILE EDIT
# ═══════════════════════════════════════════════════════════════

async def cmd_edit(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid     = update.effective_user.id
    profile = get_user(uid)
    if not profile:
        await update.message.reply_text("❌ No profile found. Use /start to create one.")
        return ConversationHandler.END

    ctx.user_data["edit_profile"] = profile.copy()
    all_al   = get_all_allergens(profile)
    al_display = " • ".join(format_allergen_display(a) for a in all_al) if all_al else "None"
    await update.message.reply_text(
        f"✏️ <b>Edit Your Profile</b>\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>Name:</b>      {profile.get('name','')}\n"
        f"🎂 <b>Age:</b>       {profile.get('age','')}\n"
        f"⚖️ <b>Weight:</b>    {profile.get('weight','')} kg\n"
        f"⚠️ <b>Allergens:</b> {al_display}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"👇 What would you like to change?",
        parse_mode=ParseMode.HTML,
        reply_markup=profile_edit_kb()
    )
    return EDIT_FIELD

async def edit_field_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data  = query.data

    if data == "edit_name":
        await query.edit_message_text("✏️ <b>Edit Name</b>\n\nType your new name:", parse_mode=ParseMode.HTML)
        return EDIT_NAME
    if data == "edit_age":
        await query.edit_message_text("🎂 <b>Edit Age</b>\n\nSelect your age range:", parse_mode=ParseMode.HTML, reply_markup=age_kb(prefix="eage"))
        return EDIT_AGE
    if data == "edit_weight":
        await query.edit_message_text("⚖️ <b>Edit Weight</b>\n\nEnter your new weight in kg:\n<i>e.g.</i> <code>70</code>", parse_mode=ParseMode.HTML)
        return EDIT_WEIGHT
    if data == "edit_allergens":
        profile  = ctx.user_data.get("edit_profile", {})
        std      = profile.get("allergens", [])
        custom   = profile.get("custom_allergens", [])
        # selected holds ALL active allergens (standard + custom) for unified toggle logic
        selected = std[:] + custom[:]
        ctx.user_data["edit_allergens"]        = selected
        ctx.user_data["edit_custom_allergens"] = custom[:]
        await query.edit_message_text(
            "⚙️ <b>Edit Your Allergens</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Tap any allergen to read about it,\nthen toggle it on or off.\n"
            "Use <b>➕ Add my own</b> to add a custom one:",
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_kb(selected, custom, prefix="eallergen", show_info=True)
        )
        return EDIT_ALLERGENS
    if data == "edit_done":
        return await _save_edit(query, ctx)
    return EDIT_FIELD

async def edit_name_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["edit_profile"]["name"] = update.message.text.strip()
    await _show_edit_menu(update.message, ctx)
    return EDIT_FIELD

async def edit_age_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    raw = query.data.replace("eage_", "").replace("_", "–")
    ctx.user_data["edit_profile"]["age"] = raw
    await query.edit_message_text(f"✅ Age updated to <b>{raw}</b>.", parse_mode=ParseMode.HTML)
    await _show_edit_menu(query.message, ctx)
    return EDIT_FIELD

async def edit_weight_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data["edit_profile"]["weight"] = int(update.message.text.strip())
        await _show_edit_menu(update.message, ctx)
    except ValueError:
        await update.message.reply_text("⚠️ Please enter a number only — e.g. <code>70</code>", parse_mode=ParseMode.HTML)
    return EDIT_FIELD

async def edit_allergen_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handles allergen callbacks during profile editing."""
    query    = update.callback_query
    await query.answer()
    data     = query.data
    selected = ctx.user_data.get("edit_allergens", [])
    custom   = ctx.user_data.get("edit_custom_allergens", [])

    # ── Back to list — MUST be before startswith checks ───────
    if data == "eallergen_back_list":
        count = len(selected) + len([c for c in custom if c not in ALLERGEN_LIST])
        await query.edit_message_text(
            f"⚙️ <b>Edit Your Allergens</b>\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{'✅ ' + str(count) + ' selected' if count else '⬜ None selected'}\n\n"
            f"Tap an allergen to read about it,\nthen toggle it on or off:",
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_kb(selected, custom, prefix="eallergen", show_info=True)
        )
        return EDIT_ALLERGENS

    # ── Add custom allergen ────────────────────────────────────
    if data == "eallergen_add_custom":
        await query.edit_message_text(
            "➕ <b>Add Your Own Allergen</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Type the name of your allergen below.\n\n"
            "<i>Examples: sesame, shellfish, mustard...</i>",
            parse_mode=ParseMode.HTML
        )
        ctx.user_data["awaiting_custom_allergen_context"] = "edit"
        return EDIT_ADD_CUSTOM_ALLERGEN

    # ── Show allergen info ─────────────────────────────────────
    if data.startswith("eallergen_info_"):
        allergen = data.replace("eallergen_info_", "")
        info     = ALLERGEN_INFO.get(allergen, "No information available.")
        await query.edit_message_text(
            info, parse_mode=ParseMode.HTML,
            reply_markup=allergen_info_view_kb(allergen, prefix="eallergen", selected=selected, in_edit=True)
        )
        return EDIT_ALLERGENS

    # ── Toggle allergen ────────────────────────────────────────
    if data.startswith("eallergen_toggle_"):
        allergen = data.replace("eallergen_toggle_", "")
        if allergen in selected: selected.remove(allergen)
        else: selected.append(allergen)
        ctx.user_data["edit_allergens"] = selected

        if allergen in ALLERGEN_LIST:
            info = ALLERGEN_INFO.get(allergen, "")
            await query.edit_message_text(
                info, parse_mode=ParseMode.HTML,
                reply_markup=allergen_info_view_kb(allergen, prefix="eallergen", selected=selected, in_edit=True)
            )
        else:
            await query.edit_message_text(
                "⚙️ <b>Edit Your Allergens</b>\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Tap an allergen to read about it,\nthen toggle it on or off:",
                parse_mode=ParseMode.HTML,
                reply_markup=allergen_kb(selected, custom, prefix="eallergen", show_info=True)
            )
        return EDIT_ALLERGENS

    # ── Done ───────────────────────────────────────────────────
    # ── Done ───────────────────────────────────────────────────
    if data == "eallergen_done":
        std_selected    = [a for a in selected if a in ALLERGEN_LIST]
        custom_selected = [a for a in selected if a not in ALLERGEN_LIST]
        ctx.user_data["edit_profile"]["allergens"]        = std_selected
        ctx.user_data["edit_profile"]["custom_allergens"] = custom_selected
        count = len(std_selected) + len(custom_selected)
        await query.edit_message_text(
            f"✅ <b>Allergens updated!</b>\n\n"
            f"{'⚠️ ' + str(count) + ' allergen(s) saved to your profile.' if count else '✅ No allergens set.'}",
            parse_mode=ParseMode.HTML
        )
        await _show_edit_menu(query.message, ctx)
        return EDIT_FIELD

    return EDIT_ALLERGENS

async def edit_receive_custom_allergen(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Receive a custom allergen typed by the user during edit."""
    allergen_name = update.message.text.strip().lower()
    selected      = ctx.user_data.get("edit_allergens", [])
    custom        = ctx.user_data.get("edit_custom_allergens", [])

    if allergen_name and allergen_name not in custom and allergen_name not in ALLERGEN_LIST:
        custom.append(allergen_name)
        ctx.user_data["edit_custom_allergens"] = custom
        selected.append(allergen_name)
        ctx.user_data["edit_allergens"] = selected
        conf = f"✅ <b>'{allergen_name.capitalize()}'</b> added!"
    elif allergen_name in ALLERGEN_LIST:
        conf = f"ℹ️ <b>'{allergen_name.capitalize()}'</b> is in the standard list — tap it to add it."
    elif allergen_name in custom:
        conf = f"ℹ️ <b>'{allergen_name.capitalize()}'</b> is already in your custom list."
    else:
        conf = "⚠️ Please enter a valid allergen name."

    await update.message.reply_text(
        f"{conf}\n\nYou can add more or tap ✅ Done when finished:",
        parse_mode=ParseMode.HTML,
        reply_markup=allergen_kb(selected, custom, prefix="eallergen", show_info=True)
    )
    return EDIT_ALLERGENS

async def _show_edit_menu(message, ctx):
    profile    = ctx.user_data.get("edit_profile", {})
    all_al     = get_all_allergens(profile)
    al_display = " • ".join(format_allergen_display(a) for a in all_al) if all_al else "None"
    await message.reply_text(
        f"✏️ <b>Edit Your Profile</b>\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>Name:</b>      {profile.get('name','')}\n"
        f"🎂 <b>Age:</b>       {profile.get('age','')}\n"
        f"⚖️ <b>Weight:</b>    {profile.get('weight','')} kg\n"
        f"⚠️ <b>Allergens:</b> {al_display}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"👇 What else would you like to change?",
        parse_mode=ParseMode.HTML,
        reply_markup=profile_edit_kb()
    )

async def _save_edit(query, ctx):
    uid     = query.from_user.id
    profile = ctx.user_data.get("edit_profile", {})
    save_user(uid, profile)
    all_al     = get_all_allergens(profile)
    al_display = " • ".join(format_allergen_display(a) for a in all_al) if all_al else "None"
    await query.edit_message_text(
        f"💾 <b>Profile Saved!</b>\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>Name:</b>      {profile.get('name','')}\n"
        f"🎂 <b>Age:</b>       {profile.get('age','')}\n"
        f"⚖️ <b>Weight:</b>    {profile.get('weight','')} kg\n"
        f"⚠️ <b>Allergens:</b> {al_display}\n"
        f"━━━━━━━━━━━━━━━━━━━━",
        parse_mode=ParseMode.HTML
    )
    return ConversationHandler.END

# ═══════════════════════════════════════════════════════════════
# PHOTO HANDLER
# ═══════════════════════════════════════════════════════════════

async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid     = update.effective_user.id
    profile = get_user(uid)

    if not profile:
        await update.message.reply_text(
            "👤 <b>Profile required!</b>\n\nPlease set up your profile first.",
            parse_mode=ParseMode.HTML,
            reply_markup=setup_menu_kb()
        )
        return

    status   = await update.message.reply_text("📸 Photo received! Starting analysis...")
    tmp_path = None

    try:
        await status.edit_text(
            "📥 <b>Step 1/3</b> — Downloading image...\n▓░░░░░░░░░ 10%",
            parse_mode=ParseMode.HTML
        )
        photo = update.message.photo[-1]
        file  = await photo.get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

        await status.edit_text(
            "🔍 <b>Step 2/3</b> — Extracting text with OCR...\n▓▓▓▓░░░░░░ 40%\n\n"
            "<i>Using fine-tuned eng_textocr model</i>",
            parse_mode=ParseMode.HTML
        )
        ocr      = run_ocr(tmp_path)
        ocr_text = ocr["text"]

        if not ocr_text.strip():
            await status.edit_text(
                "❌ <b>No text detected</b>\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Tips for a better scan:\n\n"
                "💡 Use good lighting\n"
                "📐 Hold camera straight\n"
                "🔎 Make label fill the frame\n"
                "🚫 Avoid glare or shadows",
                parse_mode=ParseMode.HTML
            )
            return

        await status.edit_text(
            "🧠 <b>Step 3/3</b> — AI analysing ingredients...\n▓▓▓▓▓▓▓░░░ 70%\n\n"
            "<i>Mistral 7B is reading the label...</i>",
            parse_mode=ParseMode.HTML
        )
        analysis = analyze_label(ocr_text, profile)

        if not analysis["is_food_label"]:
            await status.edit_text(
                f"🚫 <b>Not a food label</b>\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💬 {analysis.get('reason','')}\n\n"
                f"📸 Please send a photo of a food product label.",
                parse_mode=ParseMode.HTML
            )
            return

        # ── Allergen matching (standard + custom) ─────────────
        all_user_allergens = get_all_allergens(profile)
        found_allergens    = []
        for detected in analysis.get("allergens", []):
            for user_al in all_user_allergens:
                if user_al.lower() in detected.lower():
                    found_allergens.append(detected)
                    break

        # ── Build the structured result message ───────────────
        rating  = analysis["health_rating"]
        emoji   = RATING_EMOJI.get(rating, "⚪")
        label   = RATING_LABEL.get(rating, "Unknown")
        verdict = RATING_VERDICT.get(rating, "")

        # Health Rating block
        rating_block = (
            f"{emoji} <b>Health Rating: {label}</b>\n\n"
            f"{verdict}"
        )

        # Summary block
        summary = analysis.get("summary", "")
        summary_block = f"\n\n━━━━━━━━━━━━━━━━━━━━\n📝 <b>Why this rating?</b>\n{summary}" if summary else ""

        # Ingredients block
        ingredients = analysis.get("ingredients", [])
        if ingredients:
            ing_lines   = "\n".join(f"  • {i}" for i in ingredients[:15])
            ing_block   = f"\n\n━━━━━━━━━━━━━━━━━━━━\n🧪 <b>Ingredients</b>\n{ing_lines}"
            if len(ingredients) > 15:
                ing_block += f"\n  <i>...and {len(ingredients)-15} more</i>"
        else:
            ing_block = ""

        # Warnings block
        warnings = analysis.get("warnings", [])
        if warnings:
            w_lines   = "\n".join(f"  🚫 {w}" for w in warnings)
            w_block   = f"\n\n━━━━━━━━━━━━━━━━━━━━\n⚠️ <b>Health Warnings</b>\n{w_lines}"
        else:
            w_block = ""

        # Positives block
        positives = analysis.get("positives", [])
        if positives:
            p_lines = "\n".join(f"  ✅ {p}" for p in positives)
            p_block = f"\n\n━━━━━━━━━━━━━━━━━━━━\n💚 <b>Positives</b>\n{p_lines}"
        else:
            p_block = ""

        # Allergen alert block
        if found_allergens:
            al_lines  = "\n".join(f"  ⚠️ {a}" for a in found_allergens)
            al_block  = (
                f"\n\n━━━━━━━━━━━━━━━━━━━━\n"
                f"🚨🚨 <b>YOUR ALLERGENS DETECTED!</b> 🚨🚨\n"
                f"{al_lines}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>This product contains your allergens!</b>"
            )
        else:
            al_block = f"\n\n✅ <b>No allergens from your list detected</b>"

        msg = (
            f"✅ <b>Analysis Complete!</b>\n\n"
            f"{rating_block}"
            f"{summary_block}"
            f"{ing_block}"
            f"{w_block}"
            f"{p_block}"
            f"{al_block}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👇 <b>What would you like to do?</b>"
        )

        sessions[uid] = {
            "ocr_text": ocr_text,
            "analysis": analysis,
            "profile":  profile,
            "product":  analysis.get("summary", "")[:100],
        }

        await status.delete()
        await update.message.reply_text(
            msg[:4096],
            parse_mode=ParseMode.HTML,
            reply_markup=after_scan_kb()
        )

    except Exception as e:
        log.error(f"Photo handler error: {e}", exc_info=True)
        await status.edit_text(
            f"❌ <b>Something went wrong</b>\n\n<code>{str(e)[:200]}</code>",
            parse_mode=ParseMode.HTML
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ═══════════════════════════════════════════════════════════════
# CALLBACK HANDLER
# ═══════════════════════════════════════════════════════════════

async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid  = query.from_user.id
    data = query.data

    # ── History: view a scan ───────────────────────────────────
    if data.startswith("hist_view_"):
        index = int(data.replace("hist_view_", ""))
        scans = get_scans(uid)
        if 0 <= index < len(scans):
            s      = scans[index]
            detail = format_scan_detail(s)
            await query.message.reply_text(
                detail,
                parse_mode=ParseMode.HTML,
                reply_markup=scan_detail_kb(index)
            )
        else:
            await query.message.reply_text("❌ Scan not found. It may have been deleted.")
        return

    # ── History: delete a scan ─────────────────────────────────
    if data.startswith("hist_delete_"):
        index = int(data.replace("hist_delete_", ""))
        if delete_scan(uid, index):
            await query.edit_message_text(
                "🗑️ <b>Scan deleted successfully.</b>\n\n"
                "Tap 📊 History to view your remaining scans.",
                parse_mode=ParseMode.HTML
            )
        else:
            await query.edit_message_text("❌ Could not delete scan. It may have already been removed.")
        return

    # ── History: back to list ──────────────────────────────────
    if data == "hist_back":
        scans = get_scans(uid)
        if scans:
            await query.message.reply_text(
                f"📊 <b>Your Saved Scans</b>\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Tap any scan to view its full details:",
                parse_mode=ParseMode.HTML,
                reply_markup=history_list_kb(scans)
            )
        else:
            await query.message.reply_text(
                "📊 <b>No saved scans</b>\n\n"
                "Scan a food label and tap ⭐ Save to store it here.",
                parse_mode=ParseMode.HTML
            )
        return

    # ── Main-menu allergen info — exact match BEFORE startswith ─
    if data == "ainfo_back":
        try:
            await query.edit_message_text(
                "⚙️ <b>Allergens Guide</b>\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Tap any allergen to learn about it:",
                parse_mode=ParseMode.HTML,
                reply_markup=allergen_main_menu_kb()
            )
        except Exception:
            await query.message.reply_text(
                "⚙️ <b>Allergens Guide</b>\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Tap any allergen to learn about it:",
                parse_mode=ParseMode.HTML,
                reply_markup=allergen_main_menu_kb()
            )
        return

    if data.startswith("ainfo_"):
        allergen = data.replace("ainfo_", "")
        info     = ALLERGEN_INFO.get(allergen, "No information available.")
        await query.edit_message_text(
            info, parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to allergen list", callback_data="ainfo_back")]
            ])
        )
        return

    # ── After-scan buttons ─────────────────────────────────────
    if data == "pdf":
        session = sessions.get(uid)
        if not session:
            await query.edit_message_text("❌ Session expired. Please scan a label first.")
            return
        await query.edit_message_text(
            "📄 <b>Generating your PDF report...</b>\n<i>Just a moment ⏳</i>",
            parse_mode=ParseMode.HTML
        )
        pdf_bytes = build_pdf(
            session.get("ocr_text", ""),
            session.get("analysis", {}),
            session.get("profile", {})
        )
        await query.message.reply_document(
            document=io.BytesIO(pdf_bytes),
            filename=f"nutriscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            caption="📋 <b>NutriScan AI — Food Label Report</b>\n\n✅ Your full analysis is ready!",
            parse_mode=ParseMode.HTML
        )
        await query.edit_message_text("✅ <b>PDF sent!</b>", parse_mode=ParseMode.HTML, reply_markup=after_scan_kb())
        return

    if data == "ask_ai":
        ctx.user_data["asking_ai"] = True
        await query.message.reply_text(
            "💬 <b>Ask AI About This Product</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "💡 <b>Example questions:</b>\n\n"
            "  🏃 Is this good for weight loss?\n"
            "  🍬 How much sugar is in this?\n"
            "  📅 Can I eat this every day?\n"
            "  💉 Is this safe for diabetics?\n"
            "  👶 Is this safe for children?\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "✍️ <b>Type your question:</b>",
            parse_mode=ParseMode.HTML
        )
        return

    if data == "new_scan":
        ctx.user_data["asking_ai"] = False
        await query.message.reply_text(
            "📸 <b>Ready for a new scan!</b>\n\nSend me a food label photo 🔍",
            parse_mode=ParseMode.HTML,
            reply_markup=main_menu_kb()
        )
        return

    if data == "save":
        session = sessions.get(uid)
        if session:
            save_scan(uid, {
                "date":     datetime.now().isoformat(),
                "ocr_text": session.get("ocr_text", ""),
                "analysis": session.get("analysis", {}),
                "product":  session.get("product", "Unknown product"),
                "rating":   session.get("analysis", {}).get("health_rating", "unknown"),
            })
            await query.message.reply_text(
                "⭐ <b>Scan saved to your history!</b>\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "📊 View it anytime in the History menu.",
                parse_mode=ParseMode.HTML,
                reply_markup=main_menu_kb()
            )
        else:
            await query.message.reply_text("❌ Nothing to save. Please scan a label first.")
        return

# ═══════════════════════════════════════════════════════════════
# TEXT HANDLER
# ═══════════════════════════════════════════════════════════════

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    uid  = update.effective_user.id

    # ── AI question mode ───────────────────────────────────────
    if ctx.user_data.get("asking_ai"):
        session = sessions.get(uid)
        if not session:
            await update.message.reply_text(
                "❌ Session expired. Please scan a label first.",
                reply_markup=main_menu_kb()
            )
            ctx.user_data["asking_ai"] = False
            return
        await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
        answer = answer_question(text, session.get("analysis", {}), session.get("profile", {}))
        await update.message.reply_text(
            f"🧠 <b>AI Answer</b>\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{answer}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"<i>Ask another question or use the buttons below</i>",
            parse_mode=ParseMode.HTML,
            reply_markup=after_scan_kb()
        )
        return

    # ── Menu buttons ───────────────────────────────────────────
    if text == "🚀 Set Up Profile & Start":
        return await setup_intro(update, ctx)

    if text == "📸 Scan a Label":
        await update.message.reply_text(
            "📸 <b>Ready to scan!</b>\n\n"
            "Send me a photo of any food label\nand I'll analyse it instantly 🔍\n\n"
            "<i>Tip: well-lit photos give the best results</i>",
            parse_mode=ParseMode.HTML,
            reply_markup=main_menu_kb()
        )
        return

    if text == "👤 My Profile":
        profile = get_user(uid)
        if profile:
            all_al     = get_all_allergens(profile)
            al_display = " • ".join(format_allergen_display(a) for a in all_al) if all_al else "None"
            await update.message.reply_text(
                f"👤 <b>Your Profile</b>\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"👤 <b>Name:</b>      {profile.get('name','')}\n"
                f"🎂 <b>Age:</b>       {profile.get('age','')}\n"
                f"⚖️ <b>Weight:</b>    {profile.get('weight','')} kg\n"
                f"⚠️ <b>Allergens:</b> {al_display}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"✏️ <i>Use /edit to update your profile</i>",
                parse_mode=ParseMode.HTML
            )
        else:
            await update.message.reply_text(
                "❌ No profile found.\nPress 🚀 Set Up Profile & Start to create one.",
                reply_markup=setup_menu_kb()
            )
        return

    if text == "⚙️ Allergens":
        await update.message.reply_text(
            "⚙️ <b>Allergens Guide</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "📖 Tap any allergen to learn about\n"
            "its symptoms, risks and common sources:",
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_main_menu_kb()
        )
        return

    if text == "📊 History":
        scans = get_scans(uid)
        if scans:
            await update.message.reply_text(
                f"📊 <b>Your Saved Scans</b>\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Tap any scan below to view its\nfull analysis details:",
                parse_mode=ParseMode.HTML,
                reply_markup=history_list_kb(scans)
            )
        else:
            await update.message.reply_text(
                "📊 <b>No saved scans yet</b>\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "📸 Scan a food label and tap\n"
                "⭐ Save to History to store it here.",
                parse_mode=ParseMode.HTML
            )
        return

    if text == "❓ Help":
        await update.message.reply_text(
            "❓ <b>How to Use NutriScan AI</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "📸 <b>Step 1</b> — Send a food label photo\n"
            "🔍 <b>Step 2</b> — OCR extracts the text\n"
            "🧠 <b>Step 3</b> — AI analyses ingredients\n"
            "🟢🟡🔴 <b>Step 4</b> — Get your health rating\n"
            "🧪 <b>Step 5</b> — See ingredients listed\n"
            "🚨 <b>Step 6</b> — See your allergen alerts\n"
            "📄 <b>Step 7</b> — Download the PDF report\n"
            "💬 <b>Step 8</b> — Ask AI any question\n"
            "⭐ <b>Step 9</b> — Save to your history\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "🤖 <b>Commands:</b>\n"
            "/start — Welcome screen\n"
            "/edit  — Edit your profile\n\n"
            "💡 <b>Tip:</b> Well-lit, straight photos\ngive the best OCR results!",
            parse_mode=ParseMode.HTML,
            reply_markup=main_menu_kb()
        )
        return

    # ── Fallback ───────────────────────────────────────────────
    profile = get_user(uid)
    if profile:
        await update.message.reply_text(
            "📸 Send me a food label photo or use the menu below 👇",
            reply_markup=main_menu_kb()
        )
    else:
        await update.message.reply_text(
            "👋 Please set up your profile first!",
            reply_markup=setup_menu_kb()
        )

# ═══════════════════════════════════════════════════════════════
# BOT COMMANDS  — /help removed, only /start and /edit
# ═══════════════════════════════════════════════════════════════

async def post_init(app: Application):
    await app.bot.set_my_commands([
        BotCommand("start", "Welcome screen"),
        BotCommand("edit",  "Edit your profile"),
    ])

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    if BOT_TOKEN == "YOUR_TOKEN_HERE":
        print("❌ Set TELEGRAM_BOT_TOKEN in your .env file!")
        sys.exit(1)

    log.info("Checking Tesseract...")
    try:
        ver = pytesseract.get_tesseract_version()
        log.info(f"Tesseract version: {ver} ✅")
    except Exception as e:
        log.error(f"Tesseract not found: {e} ❌")

    log.info("Checking Ollama...")
    log.info("Ollama is running ✅" if ollama_alive() else "Ollama NOT reachable ⚠️")

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    # ── Profile setup conversation ─────────────────────────────
    setup_conv = ConversationHandler(
        entry_points=[MessageHandler(filters.Regex("^🚀 Set Up Profile & Start$"), setup_intro)],
        states={
            NAME:                 [MessageHandler(filters.TEXT & ~filters.COMMAND, get_name)],
            AGE:                  [CallbackQueryHandler(get_age,               pattern="^age_")],
            WEIGHT:               [MessageHandler(filters.TEXT & ~filters.COMMAND, get_weight)],
            ALLERGENS:            [CallbackQueryHandler(allergen_handler,      pattern="^allergen_")],
            ADD_CUSTOM_ALLERGEN:  [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_custom_allergen)],
        },
        fallbacks=[CommandHandler("start", cmd_start)],
    )

    # ── Profile edit conversation ──────────────────────────────
    edit_conv = ConversationHandler(
        entry_points=[CommandHandler("edit", cmd_edit)],
        states={
            EDIT_FIELD:               [CallbackQueryHandler(edit_field_cb,              pattern="^edit_")],
            EDIT_NAME:                [MessageHandler(filters.TEXT & ~filters.COMMAND,  edit_name_msg)],
            EDIT_AGE:                 [CallbackQueryHandler(edit_age_cb,                pattern="^eage_")],
            EDIT_WEIGHT:              [MessageHandler(filters.TEXT & ~filters.COMMAND,  edit_weight_msg)],
            EDIT_ALLERGENS:           [CallbackQueryHandler(edit_allergen_cb,           pattern="^eallergen_")],
            EDIT_ADD_CUSTOM_ALLERGEN: [MessageHandler(filters.TEXT & ~filters.COMMAND,  edit_receive_custom_allergen)],
        },
        fallbacks=[CommandHandler("edit", cmd_edit)],
    )

    app.add_handler(setup_conv)
    app.add_handler(edit_conv)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("\n" + "═" * 52)
    print("  🍎 NUTRISCAN AI BOT — RUNNING")
    print("═" * 52)
    print("  ✅ History: clickable dates → full scan details")
    print("  ✅ History: 🗑️ Delete + 🔙 Back to History")
    print("  ✅ Custom allergens: ➕ Add my own allergen")
    print("  ✅ Custom allergens trigger scan warnings")
    print("  ✅ /help removed — Help is menu button only")
    print("  ✅ /start shows friendly NutriScan AI intro")
    print("  ✅ Analysis: Rating → Summary → Ingredients → Allergens")
    print("  ✅ PDF: OCR text small + subtle at very end")
    print("═" * 52)
    print("  Press Ctrl+C to stop\n")

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()