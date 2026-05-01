import os
import io
import sys
import time
import json
import logging
import tempfile
import asyncio
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# Telegram
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardMarkup, KeyboardButton, BotCommand,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, filters,
    ContextTypes,
)
from telegram.constants import ChatAction, ParseMode


import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
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



BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
TESS_DIR = os.getenv("TESSERACT_MODEL_DIR", "")
if TESS_DIR:
    os.environ["TESSDATA_PREFIX"] = TESS_DIR

PRIMARY_LANG = "eng_textocr"
FALLBACK_LANG = "eng"


(NAME, AGE, WEIGHT, ALLERGENS,
 EDIT_FIELD, EDIT_NAME, EDIT_AGE, EDIT_WEIGHT, EDIT_ALLERGENS,
 ASK_AI) = range(10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


DB_FILE = Path(__file__).parent / "user_data.json"

def load_db():
    """Load the user database from JSON file.
     Returns:
        Database dictionary with 'users' and 'scans' keys. If the file
        does not exist, returns an empty database structure.
    """
    if DB_FILE.exists():
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return {"users": {}, "scans": {}}

def save_db(db):
    """Save the user database to JSON file.

    Args:
        db: Database dictionary to persist to disk.
    """
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def get_user(uid: int):
    """Retrieve a user's profile by Telegram user ID.

    Args:
        uid: Telegram user ID.

    Returns:
        User profile dictionary. Returns an empty dict if no profile
        exists for the given ID.
    """
    db = load_db()
    return db["users"].get(str(uid), {})

def save_user(uid: int, data: dict):
    """Save or update a user's profile.

    Args:
        uid: Telegram user ID.
        data: Profile data dictionary to store.
    """
    db = load_db()
    db["users"][str(uid)] = data
    save_db(db)

def save_scan(uid: int, scan_data: dict):
    """Append a scan result to a user's history.

    Args:
        uid: Telegram user ID.
        scan_data: Scan data containing date, ocr_text, analysis,
            product name, and health rating.
    """
    db = load_db()
    if str(uid) not in db["scans"]:
        db["scans"][str(uid)] = []
    db["scans"][str(uid)].append(scan_data)
    save_db(db)

def get_scans(uid: int):
    """Retrieve all saved scans for a user.

    Args:
        uid: Telegram user ID.

    Returns:
        List of scan data dictionaries. Returns an empty list if the
        user has no saved scans.
    """
    db = load_db()
    return db["scans"].get(str(uid), [])



RATING_EMOJI = {"healthy": "🟢", "moderate": "🟡", "unhealthy": "🔴", "unknown": "⚪"}
RATING_LABEL = {"healthy": "Healthy", "moderate": "Moderate", "unhealthy": "Unhealthy", "unknown": "Unknown"}
ALLERGEN_LIST = ["milk", "gluten", "peanut", "soy", "egg", "lactose", "nuts", "wheat"]
ALLERGEN_EMOJIS = {
    "milk": "🥛", "gluten": "🌾", "peanut": "🥜", "soy": "🫘",
    "egg": "🥚", "lactose": "🥛", "nuts": "🥜", "wheat": "🌾"
}

ALLERGEN_INFO = {
    "milk":    "🥛 <b>Milk Allergy</b>\n\nAn immune reaction to proteins in cow's milk. Symptoms include hives, wheezing, vomiting, and in severe cases anaphylaxis.\n\n<i>Commonly found in: dairy products, cheese, butter, cream, yogurt, whey.</i>",
    "gluten":  "🌾 <b>Gluten</b>\n\nCeliac disease affects ~1% of people worldwide. Gluten damages the small intestine lining. Symptoms: bloating, diarrhea, fatigue, and long-term nutrient deficiency.\n\n<i>Commonly found in: bread, pasta, cereals, beer, sauces.</i>",
    "peanut":  "🥜 <b>Peanut Allergy</b>\n\nOne of the most severe food allergies. Even trace amounts can trigger life-threatening anaphylaxis. Often lifelong.\n\n<i>Commonly found in: peanut butter, snacks, Asian cuisine, baked goods.</i>",
    "soy":     "🫘 <b>Soy Allergy</b>\n\nCommon in infants and young children. Most outgrow it by age 3–5. Symptoms: rash, itching, digestive upset.\n\n<i>Commonly found in: soy sauce, tofu, edamame, processed foods.</i>",
    "egg":     "🥚 <b>Egg Allergy</b>\n\nSecond most common food allergy in children. Reactions range from mild rash to severe anaphylaxis. Many children outgrow it.\n\n<i>Commonly found in: baked goods, mayonnaise, pasta, sauces.</i>",
    "lactose": "🥛 <b>Lactose Intolerance</b>\n\nAffects up to 65% of the world population. The body cannot digest lactose (milk sugar). Causes bloating, gas, and diarrhea — not an allergy but a digestive intolerance.\n\n<i>Commonly found in: milk, cheese, ice cream, cream-based products.</i>",
    "nuts":    "🥜 <b>Tree Nut Allergy</b>\n\nIncludes almonds, walnuts, cashews, pistachios. Often lifelong and can cause severe anaphylaxis. Distinct from peanut allergy.\n\n<i>Commonly found in: nut butters, chocolates, baked goods, trail mixes.</i>",
    "wheat":   "🌾 <b>Wheat Allergy</b>\n\nAn immune reaction to wheat proteins — different from celiac disease. Symptoms: swelling, itching, breathing difficulties, digestive issues.\n\n<i>Commonly found in: bread, pasta, cereals, soy sauce, couscous.</i>",
}

RATING_COLOR = {"healthy": "#2e7d32", "moderate": "#e65100", "unhealthy": "#b71c1c", "unknown": "#546e7a"}


_easyocr_reader = None

def _get_easy_reader():
    """Lazy-load and return the EasyOCR reader instance.

    Returns:
        EasyOCR Reader object configured for English language.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader

def enhance_for_ocr(image_path: str) -> Image.Image:
    """Preprocess an image for optimal OCR performance.

    Applies the following pipeline:
        1. Load image from path
        2. Upscale if width is below 1200 pixels
        3. Convert to grayscale
        4. Denoise using fast non-local means
        5. Apply adaptive threshold binarization

    Args:
        image_path: Filesystem path to the source image.

    Returns:
        A binarized PIL Image ready for Tesseract OCR.
    """
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        img_pil = Image.open(image_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    if w < 1200:
        scale = 1200 / w
        img_cv = cv2.resize(img_cv, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(binary)

def run_ocr(image_path: str) -> dict:
    """Extract text from an image using OCR.

    Attempts the fine-tuned Tesseract model first, falling back to the
    base English model, and finally to EasyOCR if both fail.

    Args:
        image_path: Filesystem path to the source image.

    Returns:
        Dictionary containing:
            - text: The best extracted text.
            - engine: Which engine produced the result
              ("Tesseract" or "EasyOCR").
            - tesseract: Raw Tesseract output.
            - easyocr: Raw EasyOCR output.
    """
    enhanced = enhance_for_ocr(image_path)
    tess = ""
    for lang in (PRIMARY_LANG, FALLBACK_LANG):
        try:
            for psm in (6, 7, 11, 3):
                t = pytesseract.image_to_string(enhanced, lang=lang, config=f"--psm {psm}").strip()
                if len(t) > len(tess): tess = t
        except Exception:
            pass
    try:
        reader = _get_easy_reader()
        img_arr = np.array(Image.open(image_path).convert("RGB"))
        easy = " ".join(txt for _, txt, _ in reader.readtext(img_arr)).strip()
    except Exception:
        easy = ""

    if tess and len(tess) >= len(easy or ""):
        text, engine = tess, "Tesseract"
    elif easy:
        text, engine = easy, "EasyOCR"
    elif tess:
        text, engine = tess, "Tesseract"
    else:
        text, engine = "", "none"

    return {"text": text, "engine": engine, "tesseract": tess, "easyocr": easy}

def ollama_alive():
    """Check whether the local Ollama server is reachable.

    Returns:
        True if Ollama responds to the API tags endpoint, False otherwise.
    """
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
    except:
        return False

def call_llm(prompt: str, timeout: int = 120) -> str:
    """Send a prompt to the Ollama LLM and return the response.

    Args:
        prompt: The text prompt to send.
        timeout: Request timeout in seconds.

    Returns:
        The model's response text, or an error message if the request
        fails.
    """
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }, timeout=timeout)
        return r.json()["message"]["content"].strip() if r.status_code == 200 else f"Error {r.status_code}"
    except Exception as e:
        return f"❌ LLM error: {e}"

def analyze_label(ocr_text: str, user_profile: dict = None) -> dict:
    """Analyse OCR-extracted food label text using the LLM.

    Args:
        ocr_text: Raw text extracted from the food label image.
        user_profile: Optional user profile for personalised analysis.

    Returns:
        Analysis result dictionary with keys:
            - is_food_label: Whether the text appears to be a food label.
            - reason: Explanation if not a food label.
            - health_rating: One of healthy/moderate/unhealthy/unknown.
            - summary: 2-3 sentence product summary.
            - ingredients: List of ingredient strings.
            - allergens: List of detected allergen strings.
            - nutrition: One-line nutrition summary.
            - warnings: List of health concern strings.
            - positives: List of positive aspect strings.
            - raw: Raw LLM response text.
    """
    if len(ocr_text.strip()) < 10:
        return {
            "is_food_label": False, "reason": "Not enough text was extracted.",
            "health_rating": "unknown", "summary": "", "ingredients": [],
            "allergens": [], "nutrition": "", "warnings": [], "positives": []
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
SUMMARY: <2-3 sentences>
INGREDIENTS: <comma-separated>
ALLERGENS: <comma-separated or "none">
NUTRITION: <one line summary>
WARNINGS: <comma-separated or "none">
POSITIVES: <comma-separated or "none">"""

    raw = call_llm(prompt)
    result = {
        "is_food_label": False, "reason": "Analysis failed.", "health_rating": "unknown",
        "summary": "", "ingredients": [], "allergens": [], "nutrition": "",
        "warnings": [], "positives": [], "raw": raw
    }

    for line in raw.splitlines():
        line = line.strip()
        def v(p): return line.replace(p, "").strip()
        if line.startswith("IS_FOOD_LABEL:"):   result["is_food_label"] = v("IS_FOOD_LABEL:").lower() in ("yes","true","1")
        elif line.startswith("REASON:"):         result["reason"] = v("REASON:")
        elif line.startswith("HEALTH_RATING:"):
            r = v("HEALTH_RATING:").lower()
            if r in ("healthy","moderate","unhealthy"): result["health_rating"] = r
        elif line.startswith("SUMMARY:"):        result["summary"] = v("SUMMARY:")
        elif line.startswith("INGREDIENTS:"):    result["ingredients"] = [i.strip() for i in v("INGREDIENTS:").split(",") if i.strip()]
        elif line.startswith("ALLERGENS:"):      result["allergens"] = [a.strip() for a in v("ALLERGENS:").split(",") if a.strip() and a.lower() != "none"]
        elif line.startswith("NUTRITION:"):      result["nutrition"] = v("NUTRITION:")
        elif line.startswith("WARNINGS:"):       result["warnings"] = [w.strip() for w in v("WARNINGS:").split(",") if w.strip() and w.lower() != "none"]
        elif line.startswith("POSITIVES:"):      result["positives"] = [p.strip() for p in v("POSITIVES:").split(",") if p.strip() and p.lower() != "none"]

    return result

def answer_question(question: str, analysis: dict, user_profile: dict = None) -> str:
    """Answer a user's nutrition question about a scanned product.

    Args:
        question: The user's question.
        analysis: The analysis result from analyze_label().
        user_profile: Optional user profile for context.

    Returns:
        AI-generated answer in 2-4 sentences.
    """
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



def build_pdf(ocr_text: str, analysis: dict, user_profile: dict = None) -> bytes:
    """Generate a styled PDF report for a food label analysis.

    Args:
        ocr_text: Raw OCR-extracted text.
        analysis: Analysis result dictionary.
        user_profile: Optional user profile to include.

    Returns:
        PDF file contents as raw bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    def h1(t):
        return Paragraph(t, ParagraphStyle("h1", parent=styles["Title"], fontSize=22,
                                           textColor=colors.HexColor("#1a237e"), spaceAfter=4))
    def h2(t):
        return Paragraph(t, ParagraphStyle("h2", parent=styles["Heading2"], fontSize=14,
                                           textColor=colors.HexColor("#283593"), spaceBefore=10, spaceAfter=4))
    def body(t, c="#212121"):
        return Paragraph(t, ParagraphStyle("body", parent=styles["Normal"], fontSize=11,
                                           textColor=colors.HexColor(c), leading=16))
    def hr():
        return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c5cae9"))

    rating = analysis.get("health_rating", "unknown")
    r_color = RATING_COLOR.get(rating, "#546e7a")

    story += [h1("🍎 NutriScan AI — Food Label Report"),
              body(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "#757575"),
              hr(), Spacer(1, 0.3*cm)]

    if user_profile:
        story += [h2("User Profile"),
                  body(f"Name: {user_profile.get('name','N/A')} | "
                       f"Age: {user_profile.get('age','N/A')} | "
                       f"Weight: {user_profile.get('weight','N/A')}kg"),
                  Spacer(1, 0.2*cm)]

    story += [h2("Health Rating"),
              Paragraph(f'<font color="{r_color}"><b>{rating.capitalize()}</b></font>',
                        ParagraphStyle("rating", parent=styles["Normal"], fontSize=14, leading=20)),
              Spacer(1, 0.2*cm)]

    if analysis.get("summary"):     story += [h2("Summary"),        body(analysis["summary"]),                    Spacer(1, 0.2*cm)]
    if analysis.get("ingredients"): story += [h2("Ingredients"),    body(", ".join(analysis["ingredients"])),     Spacer(1, 0.2*cm)]

    story.append(h2("Allergens"))
    if analysis.get("allergens"):
        story += [body(f"⚠ {a}", "#b71c1c") for a in analysis["allergens"]]
    else:
        story.append(body("✓ No common allergens detected", "#2e7d32"))
    story.append(Spacer(1, 0.2*cm))

    if analysis.get("nutrition"):  story += [h2("Nutrition Facts"),  body(analysis["nutrition"]),                 Spacer(1, 0.2*cm)]
    if analysis.get("warnings"):   story += [h2("Health Warnings")]  + [body(f"✗ {w}", "#b71c1c") for w in analysis["warnings"]]  + [Spacer(1, 0.2*cm)]
    if analysis.get("positives"):  story += [h2("Positive Aspects")] + [body(f"✓ {p}", "#2e7d32") for p in analysis["positives"]] + [Spacer(1, 0.2*cm)]

    story += [hr(), Spacer(1, 0.2*cm),
              h2("Extracted Text (OCR)"), body(ocr_text[:3000], "#546e7a"),
              Spacer(1, 0.5*cm), hr(),
              body("Generated by NutriScan AI • Powered by Fine-tuned Tesseract + Ollama", "#9e9e9e")]

    doc.build(story)
    return buf.getvalue()

def main_menu_kb():
    """Create the main menu reply keyboard.

    Returns:
        ReplyKeyboardMarkup with Scan, Profile, History, Allergens,
        and Help buttons.
    """
    return ReplyKeyboardMarkup([
        ["📸 Scan a Label", "👤 My Profile"],
        ["📊 History",      "⚙️ Allergens"],
        ["❓ Help"]
    ], resize_keyboard=True)

def setup_menu_kb():
    """Create the initial setup prompt keyboard.

    Returns:
        ReplyKeyboardMarkup with a single setup button.
    """
    return ReplyKeyboardMarkup([
        ["🚀 Set Up Profile & Start"]
    ], resize_keyboard=True)

def age_kb(prefix="age"):
    """Create an age selection inline keyboard.

    Args:
        prefix: Callback data prefix for distinguishing setup vs edit.

    Returns:
        InlineKeyboardMarkup with age range buttons.
    """
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Under 18",  callback_data=f"{prefix}_under18"),
         InlineKeyboardButton("18–25",     callback_data=f"{prefix}_18_25")],
        [InlineKeyboardButton("26–35",     callback_data=f"{prefix}_26_35"),
         InlineKeyboardButton("36–50",     callback_data=f"{prefix}_36_50")],
        [InlineKeyboardButton("50+",       callback_data=f"{prefix}_50plus")],
    ])

def allergen_kb(selected: list = None, prefix: str = "allergen", show_info: bool = False):
    """Create an allergen selection inline keyboard.

    Args:
        selected: List of currently selected allergen keys.
        prefix: Callback data prefix for distinguishing setup vs edit.
        show_info: If True, buttons navigate to info views instead of
            toggling directly.

    Returns:
        InlineKeyboardMarkup with allergen toggle/info buttons and a
        Done button.
    """
    if selected is None: selected = []
    buttons = []
    for a in ALLERGEN_LIST:
        emoji = ALLERGEN_EMOJIS.get(a, "•")
        mark  = "✅" if a in selected else "⬜"
        cb    = f"{prefix}_info_{a}" if show_info else f"{prefix}_{a}"
        buttons.append([InlineKeyboardButton(f"{mark} {emoji} {a.capitalize()}", callback_data=cb)])
    buttons.append([InlineKeyboardButton("✅ Done →", callback_data=f"{prefix}_done")])
    return InlineKeyboardMarkup(buttons)

def allergen_info_view_kb(allergen: str, prefix: str = "allergen", selected: list = None, in_edit: bool = False):
    """Create keyboard for viewing a single allergen's info.

    Args:
        allergen: The allergen key being viewed.
        prefix: Callback data prefix.
        selected: Currently selected allergens.
        in_edit: Whether this is in edit mode.

    Returns:
        InlineKeyboardMarkup with toggle and back buttons.
    """
    if selected is None: selected = []
    already = allergen in selected
    toggle_label = "✅ Remove from my list" if already else "⬜ Add to my list"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(toggle_label,        callback_data=f"{prefix}_toggle_{allergen}")],
        [InlineKeyboardButton("🔙 Back to list",   callback_data=f"{prefix}_back_list")],
    ])

def allergen_main_menu_kb():
    buttons = []
    for a in ALLERGEN_LIST:
        emoji = ALLERGEN_EMOJIS.get(a, "•")
        buttons.append([InlineKeyboardButton(f"{emoji} {a.capitalize()}", callback_data=f"ainfo_{a}")])
    return InlineKeyboardMarkup(buttons)

def after_scan_kb():
    """Create the main allergen information menu keyboard.

    Returns:
        InlineKeyboardMarkup with buttons for each allergen.
    """
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 Download PDF", callback_data="pdf"),
         InlineKeyboardButton("💬 Ask AI",       callback_data="ask_ai")],
        [InlineKeyboardButton("🔄 New Scan",     callback_data="new_scan"),
         InlineKeyboardButton("⭐ Save",          callback_data="save")],
    ])

def profile_edit_kb():
    """Create the profile editing menu keyboard.

    Returns:
        InlineKeyboardMarkup with edit options for each profile field.
    """
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✏️ Edit Name",      callback_data="edit_name")],
        [InlineKeyboardButton("✏️ Edit Age",       callback_data="edit_age")],
        [InlineKeyboardButton("✏️ Edit Weight",    callback_data="edit_weight")],
        [InlineKeyboardButton("⚙️ Edit Allergens", callback_data="edit_allergens")],
        [InlineKeyboardButton("✅ Save & Close",   callback_data="edit_done")],
    ])


sessions: Dict[int, dict] = {}



async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command.

    Shows a welcome message for new users or a welcome-back message
    for returning users.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        ConversationHandler.END to indicate no further state is expected.
    """
    user    = update.effective_user
    uid     = user.id
    profile = get_user(uid)

    if not profile:
        await update.message.reply_text(
            f"👋 Hello, <b>{user.first_name}</b>! Welcome to <b>NutriScan AI</b>.\n\n"
            f"Press the button below to get started 👇",
            parse_mode=ParseMode.HTML,
            reply_markup=setup_menu_kb()
        )
    else:
        await update.message.reply_text(
            f"👋 Welcome back, <b>{profile.get('name', user.first_name)}</b>!\n\n"
            f"📸 Send me a food label photo to scan:",
            parse_mode=ParseMode.HTML,
            reply_markup=main_menu_kb()
        )
    return ConversationHandler.END

async def setup_intro(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Begin the profile setup flow with an introduction.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        NAME state for the conversation handler.
    """
    await update.message.reply_text(
        "🍎 <b>Hi! I'm NutriScan AI.</b>\n\n"
        "I can scan any food label photo and:\n"
        "  • Extract all text using smart OCR\n"
        "  • Identify ingredients & allergens\n"
        "  • Give you a health rating (🟢 Healthy / 🟡 Moderate / 🔴 Unhealthy)\n"
        "  • Generate a PDF report\n"
        "  • Answer your nutrition questions with AI\n\n"
        "Let's set up your profile first so I can give you personalised advice.\n\n"
        "👤 <b>What's your name?</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardRemove()
    )
    return NAME

async def get_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Collect the user's name during profile setup.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        AGE state for the conversation handler.
    """
    ctx.user_data["name"] = update.message.text.strip()
    await update.message.reply_text(
        f"Nice to meet you, <b>{ctx.user_data['name']}</b>! 🎉\n\nHow old are you?",
        parse_mode=ParseMode.HTML,
        reply_markup=age_kb()
    )
    return AGE

async def get_age(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Collect the user's age during profile setup.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        WEIGHT state for the conversation handler.
    """
    query = update.callback_query
    await query.answer()
    raw = query.data.replace("age_", "").replace("_", "–")
    ctx.user_data["age"] = raw
    await query.edit_message_text(
        f"✅ Age: <b>{raw}</b>\n\nWhat's your weight in kg?",
        parse_mode=ParseMode.HTML
    )
    return WEIGHT

async def get_weight(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Collect the user's weight during profile setup.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        ALLERGENS state for the conversation handler, or WEIGHT state
        if the input was invalid.
    """
    try:
        ctx.user_data["weight"] = int(update.message.text.strip())
    except ValueError:
        await update.message.reply_text("Please enter a number (e.g. 70):")
        return WEIGHT

    ctx.user_data["allergens"] = []
    await update.message.reply_text(
        f"✅ Weight: <b>{ctx.user_data['weight']} kg</b>\n\n"
        f"Do you have any food allergies?\n"
        f"Tap an allergy to read about it, then add it to your list.",
        parse_mode=ParseMode.HTML,
        reply_markup=allergen_kb([], prefix="allergen", show_info=True)
    )
    return ALLERGENS

async def allergen_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle all allergen-related callbacks during profile setup.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        ALLERGENS state, or ConversationHandler.END when Done is pressed.
    """
    query    = update.callback_query
    await query.answer()
    data     = query.data
    selected = ctx.user_data.get("allergens", [])

    if data == "allergen_back_list":
        await query.edit_message_text(
            "Do you have any food allergies?\nTap an allergy to read about it, then add it to your list.",
            reply_markup=allergen_kb(selected, prefix="allergen", show_info=True)
        )
        return ALLERGENS

    
    if data.startswith("allergen_info_"):
        allergen = data.replace("allergen_info_", "")
        info     = ALLERGEN_INFO.get(allergen, "No information available.")
        await query.edit_message_text(
            info,
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_info_view_kb(allergen, prefix="allergen", selected=selected)
        )
        return ALLERGENS

    if data.startswith("allergen_toggle_"):
        allergen = data.replace("allergen_toggle_", "")
        if allergen in selected:
            selected.remove(allergen)
        else:
            selected.append(allergen)
        ctx.user_data["allergens"] = selected
        info = ALLERGEN_INFO.get(allergen, "")
        await query.edit_message_text(
            info,
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_info_view_kb(allergen, prefix="allergen", selected=selected)
        )
        return ALLERGENS

    if data == "allergen_done":
        uid     = query.from_user.id
        profile = {
            "name":      ctx.user_data.get("name", ""),
            "age":       ctx.user_data.get("age", ""),
            "weight":    ctx.user_data.get("weight", ""),
            "allergens": selected,
            "created":   datetime.now().isoformat(),
        }
        save_user(uid, profile)
        al_list = ", ".join(selected) if selected else "none"
        await query.edit_message_text(
            f"✅ <b>Profile saved!</b>\n\n"
            f"👤 Name: {profile['name']}\n"
            f"🎂 Age: {profile['age']}\n"
            f"⚖️ Weight: {profile['weight']} kg\n"
            f"⚠️ Allergens: {al_list}\n\n"
            f"📸 Now send me a food label photo!",
            parse_mode=ParseMode.HTML
        )
        await query.message.reply_text("Use the menu below:", reply_markup=main_menu_kb())
        return ConversationHandler.END

    return ALLERGENS



async def cmd_edit(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle the /edit command to start profile editing.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        EDIT_FIELD state for the conversation handler, or
        ConversationHandler.END if no profile exists.
    """
    uid     = update.effective_user.id
    profile = get_user(uid)
    if not profile:
        await update.message.reply_text("No profile found. Use /start to create one.")
        return ConversationHandler.END

    ctx.user_data["edit_profile"] = profile.copy()
    al = ", ".join(profile.get("allergens", [])) or "none"
    await update.message.reply_text(
        f"✏️ <b>Edit Your Profile</b>\n\n"
        f"👤 Name:      {profile.get('name','')}\n"
        f"🎂 Age:       {profile.get('age','')}\n"
        f"⚖️ Weight:    {profile.get('weight','')} kg\n"
        f"⚠️ Allergens: {al}\n\n"
        f"What would you like to change?",
        parse_mode=ParseMode.HTML,
        reply_markup=profile_edit_kb()
    )
    return EDIT_FIELD

async def edit_field_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle profile edit field selection callbacks.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        Appropriate edit state or ConversationHandler.END.
    """
    query = update.callback_query
    await query.answer()
    data  = query.data

    if data == "edit_name":
        await query.edit_message_text("✏️ Enter your new name:")
        return EDIT_NAME

    if data == "edit_age":
        await query.edit_message_text("Select your new age:", reply_markup=age_kb(prefix="eage"))
        return EDIT_AGE

    if data == "edit_weight":
        await query.edit_message_text("✏️ Enter your new weight (kg):")
        return EDIT_WEIGHT

    if data == "edit_allergens":
        profile  = ctx.user_data.get("edit_profile", {})
        selected = profile.get("allergens", [])
        ctx.user_data["edit_allergens"] = selected[:]
        await query.edit_message_text(
            "⚙️ <b>Edit Your Allergens</b>\n\nTap an allergen to read about it, then toggle it.",
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_kb(selected, prefix="eallergen", show_info=True)
        )
        return EDIT_ALLERGENS

    if data == "edit_done":
        return await _save_edit(query, ctx)

    return EDIT_FIELD

async def edit_name_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Save the edited name and return to edit menu.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        EDIT_FIELD state.
    """
    ctx.user_data["edit_profile"]["name"] = update.message.text.strip()
    await _show_edit_menu(update.message, ctx)
    return EDIT_FIELD

async def edit_age_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Save the edited age and return to edit menu.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        EDIT_FIELD state.
    """
    query = update.callback_query
    await query.answer()
    raw = query.data.replace("eage_", "").replace("_", "–")
    ctx.user_data["edit_profile"]["age"] = raw
    await query.edit_message_text(f"✅ Age updated to <b>{raw}</b>.", parse_mode=ParseMode.HTML)
    await _show_edit_menu_msg(query.message, ctx)
    return EDIT_FIELD

async def edit_weight_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Save the edited weight and return to edit menu.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        EDIT_FIELD state, or EDIT_WEIGHT if input is invalid.
    """
    try:
        ctx.user_data["edit_profile"]["weight"] = int(update.message.text.strip())
        await _show_edit_menu(update.message, ctx)
    except ValueError:
        await update.message.reply_text("Please enter a number (e.g. 70):")
    return EDIT_FIELD

async def edit_allergen_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle allergen-related callbacks during profile editing.

    Args:
        update: Telegram update object.
        ctx: Callback context.

    Returns:
        EDIT_ALLERGENS or EDIT_FIELD state.
    """
    """Handles allergen callbacks during profile editing."""
    query    = update.callback_query
    await query.answer()
    data     = query.data
    selected = ctx.user_data.get("edit_allergens", [])

    if data == "eallergen_back_list":
        await query.edit_message_text(
            "⚙️ <b>Edit Your Allergens</b>\n\nTap an allergen to read about it, then toggle it.",
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_kb(selected, prefix="eallergen", show_info=True)
        )
        return EDIT_ALLERGENS

    if data.startswith("eallergen_info_"):
        allergen = data.replace("eallergen_info_", "")
        info     = ALLERGEN_INFO.get(allergen, "No information available.")
        await query.edit_message_text(
            info,
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_info_view_kb(allergen, prefix="eallergen", selected=selected, in_edit=True)
        )
        return EDIT_ALLERGENS

    if data.startswith("eallergen_toggle_"):
        allergen = data.replace("eallergen_toggle_", "")
        if allergen in selected:
            selected.remove(allergen)
        else:
            selected.append(allergen)
        ctx.user_data["edit_allergens"] = selected
        info = ALLERGEN_INFO.get(allergen, "")
        await query.edit_message_text(
            info,
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_info_view_kb(allergen, prefix="eallergen", selected=selected, in_edit=True)
        )
        return EDIT_ALLERGENS

    if data == "eallergen_done":
        ctx.user_data["edit_profile"]["allergens"] = selected
        await query.edit_message_text("✅ Allergens updated!")
        await _show_edit_menu_msg(query.message, ctx)
        return EDIT_FIELD

    return EDIT_ALLERGENS

async def _show_edit_menu(message, ctx):
    """Display the edit profile menu with current values.

    Args:
        message: Telegram message object to reply to.
        ctx: Callback context containing edit_profile data.
    """
    profile = ctx.user_data.get("edit_profile", {})
    al      = ", ".join(profile.get("allergens", [])) or "none"
    await message.reply_text(
        f"✏️ <b>Edit Your Profile</b>\n\n"
        f"👤 Name:      {profile.get('name','')}\n"
        f"🎂 Age:       {profile.get('age','')}\n"
        f"⚖️ Weight:    {profile.get('weight','')} kg\n"
        f"⚠️ Allergens: {al}\n\n"
        f"What else would you like to change?",
        parse_mode=ParseMode.HTML,
        reply_markup=profile_edit_kb()
    )

async def _show_edit_menu_msg(message, ctx):
    """Wrapper for _show_edit_menu when called from a message context.

    Args:
        message: Telegram message object to reply to.
        ctx: Callback context containing edit_profile data.
    """
    await _show_edit_menu(message, ctx)

async def _save_edit(query, ctx):
    """Save the edited profile and exit the edit conversation.

    Args:
        query: Telegram callback query object.
        ctx: Callback context containing edit_profile data.

    Returns:
        ConversationHandler.END.
    """
    uid     = query.from_user.id
    profile = ctx.user_data.get("edit_profile", {})
    save_user(uid, profile)
    al = ", ".join(profile.get("allergens", [])) or "none"
    await query.edit_message_text(
        f"✅ <b>Profile Saved!</b>\n\n"
        f"👤 Name:      {profile.get('name','')}\n"
        f"🎂 Age:       {profile.get('age','')}\n"
        f"⚖️ Weight:    {profile.get('weight','')} kg\n"
        f"⚠️ Allergens: {al}",
        parse_mode=ParseMode.HTML
    )
    return ConversationHandler.END



async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Process a photo message: OCR, analyse, and return results.

    Downloads the photo, runs OCR, sends the text to the LLM for
    analysis, checks for user allergens, and displays the result.

    Args:
        update: Telegram update object containing the photo.
        ctx: Callback context.
    """
    uid 
    uid     = update.effective_user.id
    profile = get_user(uid)

    if not profile:
        await update.message.reply_text(
            "👤 Please set up your profile first!\n\nPress 🚀 Set Up Profile & Start",
            reply_markup=setup_menu_kb()
        )
        return

    status   = await update.message.reply_text("📸 Photo received...")
    tmp_path = None

    try:
        await status.edit_text("📥 Downloading image...")
        photo = update.message.photo[-1]
        file  = await photo.get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

        await status.edit_text("🔍 Extracting text with OCR...")
        ocr      = run_ocr(tmp_path)
        ocr_text = ocr["text"]

        if not ocr_text.strip():
            await status.edit_text("❌ No text found. Please try a clearer, well-lit photo.")
            return

        await status.edit_text("🧠 Analysing ingredients with AI...")
        analysis = analyze_label(ocr_text, profile)

        if not analysis["is_food_label"]:
            await status.edit_text(
                f"🚫 This doesn't look like a food label.\n\n{analysis.get('reason','')}\n\n"
                f"Please send a photo of a food product label."
            )
            return

        # Allergen warning
        user_allergens  = profile.get("allergens", [])
        found_allergens = [a for a in analysis.get("allergens", [])
                           for ua in user_allergens if ua.lower() in a.lower()]

        allergen_warning = ""
        if found_allergens:
            allergen_warning = (
                "\n\n🚨 <b>YOUR ALLERGENS DETECTED:</b>\n"
                + "\n".join(f"  ⚠️ {a}" for a in found_allergens)
            )

        rating = analysis["health_rating"]
        emoji  = RATING_EMOJI.get(rating, "⚪")
        label  = RATING_LABEL.get(rating, "Unknown")

        msg = (
            f"✅ <b>Analysis Complete!</b>\n\n"
            f"{emoji} <b>Health Rating: {label}</b>\n\n"
            f"📝 <b>Summary</b>\n{analysis.get('summary','')}"
            f"{allergen_warning}\n\n"
            f"👇 What would you like to do?"
        )

        sessions[uid] = {
            "ocr_text": ocr_text,
            "analysis": analysis,
            "profile":  profile,
            "product":  analysis.get("summary", "")[:100],
        }

        await status.delete()
        await update.message.reply_text(
            msg[:4000],
            parse_mode=ParseMode.HTML,
            reply_markup=after_scan_kb()
        )

    except Exception as e:
        log.error(f"Photo handler error for user {uid}: {e}")
        await status.edit_text(f"❌ Something went wrong: {str(e)[:200]}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)



async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle all inline button callbacks.

    Routes callbacks for allergen info, PDF generation, AI questions,
    new scan, and save actions.

    Args:
        update: Telegram update object.
        ctx: Callback context.
    """
    query = update.callback_query
    await query.answer()
    uid  = query.from_user.id
    data = query.data

    if data == "ainfo_back":
        try:
            await query.edit_message_text(
                "⚙️ <b>Allergens Guide</b>\n\nTap any allergen to learn about it:",
                parse_mode=ParseMode.HTML,
                reply_markup=allergen_main_menu_kb()
            )
        except Exception:
            await query.message.reply_text(
                "⚙️ <b>Allergens Guide</b>\n\nTap any allergen to learn about it:",
                parse_mode=ParseMode.HTML,
                reply_markup=allergen_main_menu_kb()
            )
        return

    if data.startswith("ainfo_"):
        allergen = data.replace("ainfo_", "")
        info     = ALLERGEN_INFO.get(allergen, "No information available.")
        await query.edit_message_text(
            info,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to allergen list", callback_data="ainfo_back")]
            ])
        )
        return

    if data == "pdf":
        session = sessions.get(uid)
        if not session:
            await query.edit_message_text("❌ Session expired. Please scan a label first.")
            return
        await query.edit_message_text("📄 Generating your PDF report...")
        pdf_bytes = build_pdf(
            session.get("ocr_text", ""),
            session.get("analysis", {}),
            session.get("profile", {})
        )
        await query.message.reply_document(
            document=io.BytesIO(pdf_bytes),
            filename=f"nutriscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            caption="📋 NutriScan AI — Food Label Report"
        )
        await query.edit_message_text("✅ PDF sent!", reply_markup=after_scan_kb())
        return

    if data == "ask_ai":
        ctx.user_data["asking_ai"] = True
        await query.message.reply_text(
            "💬 <b>Ask AI About This Product</b>\n\n"
            "Examples:\n"
            "  • Is this good for weight loss?\n"
            "  • How much sugar is in this?\n"
            "  • Can I eat this every day?\n"
            "  • Is this safe for diabetics?\n\n"
            "✍️ Type your question and I'll answer:",
            parse_mode=ParseMode.HTML
        )
        return

    if data == "new_scan":
        ctx.user_data["asking_ai"] = False
        await query.message.reply_text("📸 Send me a new food label photo!", reply_markup=main_menu_kb())
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
                "⭐ <b>Scan saved to your history!</b>\n\nView it anytime in 📊 History.",
                parse_mode=ParseMode.HTML,
                reply_markup=main_menu_kb()
            )
        else:
            await query.message.reply_text("❌ Nothing to save. Please scan a label first.")
        return


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle all text messages including menu navigation and AI chat.

    Args:
        update: Telegram update object.
        ctx: Callback context.
    """
    text = update.message.text.strip()
    uid  = update.effective_user.id

    # ── AI question mode ─────────────────────────────────────
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
            f"💡 <b>AI Answer</b>\n\n{answer}\n\n"
            f"<i>You can ask another question, or use the buttons below:</i>",
            parse_mode=ParseMode.HTML,
            reply_markup=after_scan_kb()
        )
        return

    if text == "🚀 Set Up Profile & Start":
        return await setup_intro(update, ctx)

   
    if text == "📸 Scan a Label":
        await update.message.reply_text("📸 Send me a food label photo!", reply_markup=main_menu_kb())
        return

    if text == "👤 My Profile":
        profile = get_user(uid)
        if profile:
            al = ", ".join(profile.get("allergens", [])) or "none"
            await update.message.reply_text(
                f"👤 <b>Your Profile</b>\n\n"
                f"Name:      {profile.get('name','')}\n"
                f"Age:       {profile.get('age','')}\n"
                f"Weight:    {profile.get('weight','')} kg\n"
                f"Allergens: {al}\n\n"
                f"<i>Use /edit to update your profile.</i>",
                parse_mode=ParseMode.HTML
            )
        else:
            await update.message.reply_text(
                "No profile found. Press 🚀 Set Up Profile & Start to create one.",
                reply_markup=setup_menu_kb()
            )
        return


    if text == "⚙️ Allergens":
        await update.message.reply_text(
            "⚙️ <b>Allergens Guide</b>\n\nTap any allergen to learn about it:",
            parse_mode=ParseMode.HTML,
            reply_markup=allergen_main_menu_kb()
        )
        return

   
    if text == "📊 History":
        scans = get_scans(uid)
        if scans:
            lines = []
            for i, s in enumerate(reversed(scans[-10:]), 1):
                date    = s.get("date", "")[:10]
                rating  = s.get("rating", s.get("analysis", {}).get("health_rating", "unknown"))
                product = s.get("product", s.get("analysis", {}).get("summary", "Unknown product"))[:60]
                emoji   = RATING_EMOJI.get(rating, "⚪")
                lines.append(f"{i}. 📅 {date}  {emoji}  {product}")
            await update.message.reply_text(
                "📊 <b>Your Saved Scans</b>\n\n" + "\n\n".join(lines),
                parse_mode=ParseMode.HTML
            )
        else:
            await update.message.reply_text(
                "📊 No saved scans yet.\n\n"
                "Scan a food label and press ⭐ Save to keep it here."
            )
        return

    if text == "❓ Help":
        await update.message.reply_text(
            "📸 <b>How to Use NutriScan AI</b>\n\n"
            "<b>Step 1:</b> Send a photo of any food label\n"
            "<b>Step 2:</b> OCR extracts all the text automatically\n"
            "<b>Step 3:</b> AI analyses ingredients & allergens\n"
            "<b>Step 4:</b> You get a health rating (🟢/🟡/🔴)\n"
            "<b>Step 5:</b> Download a PDF report\n"
            "<b>Step 6:</b> Ask AI any nutrition question\n"
            "<b>Step 7:</b> Save scans to your History\n\n"
            "<b>Commands:</b>\n"
            "/start — Welcome screen\n"
            "/edit  — Edit your profile",
            parse_mode=ParseMode.HTML,
            reply_markup=main_menu_kb()
        )
        return


    profile = get_user(uid)
    if profile:
        await update.message.reply_text(
            "📸 Send me a food label photo, or use the menu below.",
            reply_markup=main_menu_kb()
        )
    else:
        await update.message.reply_text(
            "👋 Please set up your profile first!",
            reply_markup=setup_menu_kb()
        )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle the /help command by delegating to the Help menu handler.

    Args:
        update: Telegram update object.
        ctx: Callback context.
    """

    update.message.text = "❓ Help"
    await handle_text(update, ctx)



async def post_init(app: Application):
    """Register bot commands in the Telegram menu after startup.

    Args:
        app: The Telegram Application instance.
    """
    await app.bot.set_my_commands([
        BotCommand("start", "Welcome screen"),
        BotCommand("edit",  "Edit your profile"),
        BotCommand("help",  "How to use NutriScan AI"),
    ])



def main():
    """Build and run the Telegram bot application.

    Sets up all conversation handlers, command handlers, and message
    handlers, then starts polling for updates.
    """
    if BOT_TOKEN == "YOUR_TOKEN_HERE":
        print("❌ Set TELEGRAM_BOT_TOKEN in your .env file!")
        sys.exit(1)

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    setup_conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^🚀 Set Up Profile & Start$"), setup_intro)
        ],
        states={
            NAME:      [MessageHandler(filters.TEXT & ~filters.COMMAND, get_name)],
            AGE:       [CallbackQueryHandler(get_age, pattern="^age_")],
            WEIGHT:    [MessageHandler(filters.TEXT & ~filters.COMMAND, get_weight)],
            ALLERGENS: [CallbackQueryHandler(allergen_handler, pattern="^allergen_")],
        },
        fallbacks=[CommandHandler("start", cmd_start)],
    )

    edit_conv = ConversationHandler(
        entry_points=[CommandHandler("edit", cmd_edit)],
        states={
            EDIT_FIELD:     [CallbackQueryHandler(edit_field_cb,    pattern="^edit_")],
            EDIT_NAME:      [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_name_msg)],
            EDIT_AGE:       [CallbackQueryHandler(edit_age_cb,      pattern="^eage_")],
            EDIT_WEIGHT:    [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_weight_msg)],
            EDIT_ALLERGENS: [CallbackQueryHandler(edit_allergen_cb, pattern="^eallergen_")],
        },
        fallbacks=[CommandHandler("edit", cmd_edit)],
    )

    app.add_handler(setup_conv)
    app.add_handler(edit_conv)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("\n" + "═" * 50)
    print("  🍎 NUTRISCAN AI BOT — RUNNING")
    print("═" * 50)
    print("  Fixes applied:")
    print("  ✅ 1. Set Up Profile & Start → NutriScan AI intro")
    print("  ✅ 2. Edit: only Name, Age, Weight (+ Allergens)")
    print("  ✅ 3. Allergens in edit show descriptions")
    print("  ✅ 4. ainfo_back checked BEFORE startswith — list always shows")
    print("  ✅ 5. Same fix applied in setup & edit allergen flows")
    print("  ✅ 6. Saved scans appear correctly in History")
    print("  ✅ 7. AI chat messages stay visible in chat")
    print("  ✅ 8. Help menu cleaned up — no duplicate text")
    print("═" * 50)
    print("  Press Ctrl+C to stop\n")

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()


