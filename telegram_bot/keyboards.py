# telegram_bot/keyboards.py
"""
Keyboard layouts for the Telegram bot
"""

from telegram import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton


def main_keyboard() -> ReplyKeyboardMarkup:
    """Main menu keyboard."""
    keyboard = [
        [KeyboardButton("📷 Send photo for OCR")],
        [KeyboardButton("⚙️ Change mode"), KeyboardButton("❓ Help")],
        [KeyboardButton("📊 About")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def mode_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard for selecting output mode."""
    keyboard = [
        [InlineKeyboardButton("📄 Full output",      callback_data="mode_full")],
        [InlineKeyboardButton("📝 Text only",        callback_data="mode_text")],
        [InlineKeyboardButton("💬 Summary only",     callback_data="mode_summary")],
    ]
    return InlineKeyboardMarkup(keyboard)