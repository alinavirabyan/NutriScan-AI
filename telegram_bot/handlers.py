# telegram_bot/handlers.py
"""
Message and command handlers for the Telegram bot
"""

import os
import logging
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes
from telegram_bot.keyboards import main_keyboard, mode_keyboard

logger = logging.getLogger(__name__)

TEMP_DIR = Path("./temp_images")
TEMP_DIR.mkdir(exist_ok=True)

# User settings store
user_settings: dict = {}


def get_settings(user_id: int) -> dict:
    if user_id not in user_settings:
        user_settings[user_id] = {"mode": "full"}
    return user_settings[user_id]


# ── Commands ──────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"👋 Hello {user.first_name}!\n\n"
        f"I'm an OCR bot powered by:\n"
        f"  🔍 Tesseract (fine-tuned)\n"
        f"  🤖 EasyOCR\n"
        f"  📝 Mistral AI summarization\n\n"
        f"Send me any photo with text and I'll extract + summarize it!\n\n"
        f"Commands: /help /mode /about",
        reply_markup=main_keyboard()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *How to use:*\n\n"
        "1. Send any photo containing text\n"
        "2. Bot extracts text with Tesseract + EasyOCR\n"
        "3. Mistral AI summarizes the result\n\n"
        "💡 *Tips:*\n"
        "• Good lighting = better accuracy\n"
        "• Keep text in focus\n"
        "• Horizontal text works best\n\n"
        "⚙️ Use /mode to change output format",
        parse_mode="Markdown"
    )


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *OCR Telegram Bot — Diploma Project*\n\n"
        "*Tech stack:*\n"
        "• Tesseract OCR (fine-tuned on TextOCR)\n"
        "• EasyOCR (deep learning)\n"
        "• Mistral AI (summarization)\n"
        "• Image preprocessing pipeline\n\n"
        "*Dataset:* TextOCR — 21,778 images\n"
        "*Model:* CER improved 21.6%, WER improved 31.3%",
        parse_mode="Markdown"
    )


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings = get_settings(update.effective_user.id)
    current  = settings["mode"]
    await update.message.reply_text(
        f"Current mode: *{current}*\n\nSelect new mode:",
        parse_mode="Markdown",
        reply_markup=mode_keyboard()
    )


# ── Photo handler ─────────────────────────────────────────────

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id  = update.effective_user.id
    settings = get_settings(user_id)

    msg = await update.message.reply_text(
        "🔄 Processing image...\n"
        "• Downloading\n• Preprocessing\n• Running OCR\n• Summarizing"
    )

    try:
        # Download photo
        photo    = update.message.photo[-1]
        file     = await context.bot.get_file(photo.file_id)
        img_path = str(TEMP_DIR / f"{user_id}_{photo.file_id}.jpg")
        await file.download_to_drive(img_path)

        # Run pipeline
        from ocr.pipeline import run_pipeline
        result = run_pipeline(img_path, verbose=False)

        # Format and send
        response = _format(result, settings["mode"])
        await msg.delete()
        await update.message.reply_text(response, parse_mode="Markdown")

        # Cleanup
        os.remove(img_path)

    except Exception as e:
        logger.error(f"Error: {e}")
        await msg.edit_text(f"❌ Error: {str(e)[:200]}\n\nPlease try again.")


# ── Text handler ──────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()
    if "help" in text or "❓" in text:
        await help_command(update, context)
    elif "about" in text or "📊" in text:
        await about_command(update, context)
    elif "mode" in text or "⚙️" in text:
        await mode_command(update, context)
    else:
        await update.message.reply_text(
            "📷 Please send a photo with text!\nUse /help for instructions."
        )


# ── Callback for inline keyboard ──────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query    = update.callback_query
    user_id  = query.from_user.id
    settings = get_settings(user_id)

    mode_map = {
        "mode_full":    "full",
        "mode_text":    "text_only",
        "mode_summary": "summary_only",
    }

    if query.data in mode_map:
        settings["mode"] = mode_map[query.data]
        await query.answer(f"Mode set to: {settings['mode']}")
        await query.edit_message_text(
            f"✅ Mode changed to: *{settings['mode']}*",
            parse_mode="Markdown"
        )


# ── Response formatter ────────────────────────────────────────

def _format(result: dict, mode: str) -> str:
    text    = result.get("combined_text", "").strip()
    summary = result.get("summary", "").strip()
    timing  = result.get("timing", {})

    if not text:
        return "❌ No text detected. Try a clearer photo."

    if mode == "text_only":
        return f"📄 *Extracted text:*\n```\n{text[:2000]}\n```"

    if mode == "summary_only":
        return f"📝 *Summary:*\n{summary or text[:500]}"

    # Full mode
    parts = [f"📄 *Extracted text:*\n```\n{text[:800]}\n```"]
    if summary:
        parts.append(f"\n📝 *Summary:*\n{summary}")
    parts.append(
        f"\n⏱ Total: {timing.get('total', 0):.1f}s"
    )
    return "\n".join(parts)