# # telegram_bot/bot.py
# """
# Food Label OCR Bot with Ingredient Extraction and LLM Summarization
# """

# import os
# import sys
# import time
# import logging
# import json
# import requests
# import re
# from pathlib import Path
# from datetime import datetime
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import simpleSplit

# # Add project root to path
# sys.path.append(str(Path(__file__).parent.parent))

# from telegram import Update
# from telegram.ext import (
#     Application, CommandHandler, MessageHandler,
#     filters, ContextTypes
# )
# from PIL import Image
# import pytesseract

# # Import your enhancement module
# from ocr.image_enhanced import ImageEnhancer

# # ============================================================
# # CONFIGURATION
# # ============================================================

# BOT_TOKEN = "8418376380:AAG_NBpcr87ulQ6ESxM0LyR1KOAM8Zrf6Gk"

# # Model paths
# PROJECT_ROOT = Path(__file__).parent.parent
# MODEL_PATH = PROJECT_ROOT / "models" / "tesseract" / "eng_textocr.traineddata"

# # OCR settings
# TESSERACT_LANG = "eng_textocr"
# FALLBACK_LANG = "eng"

# # Image enhancement
# ENHANCE_IMAGE = True
# SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# # LLM Settings
# USE_LLM = True
# OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "mistral"

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# def escape_html(text):
#     """Escape special characters for HTML"""
#     if not text:
#         return ""
#     # Replace problematic characters
#     text = text.replace('&', '&amp;')
#     text = text.replace('<', '&lt;')
#     text = text.replace('>', '&gt;')
#     text = text.replace('"', '&quot;')
#     return text


# class FoodLabelOCRBot:
#     """Telegram Bot for Food Label OCR with LLM Summarization"""
    
#     def __init__(self):
#         self.enhancer = ImageEnhancer()
#         self.model_name = TESSERACT_LANG
#         self.fallback = FALLBACK_LANG
#         self.stats = {"total": 0, "success": 0, "failed": 0, "models": {}}
#         self._check_model()
#         self._check_ollama()
    
#     def _check_model(self):
#         if MODEL_PATH.exists():
#             mb = MODEL_PATH.stat().st_size / 1024 / 1024
#             logger.info(f"✅ Model found: {MODEL_PATH} ({mb:.1f}MB)")
#             return True
#         logger.warning(f"⚠️ Model not found at {MODEL_PATH}, using fallback")
#         return False
    
#     def _check_ollama(self):
#         """Check if Ollama is running"""
#         try:
#             response = requests.get("http://localhost:11434/api/tags", timeout=2)
#             if response.status_code == 200:
#                 models = response.json().get("models", [])
#                 if models:
#                     logger.info(f"✅ Ollama running with models: {[m['name'] for m in models]}")
#                     return True
#             return False
#         except:
#             logger.warning("⚠️ Ollama not running. LLM features disabled.")
#             return False
    
#     def extract_ingredients(self, text):
#         """Extract ingredients list from OCR text"""
#         text_lower = text.lower()
#         ingredients = []
        
#         keywords = ["ingredients", "ingredient", "contains", "components", "composition"]
#         for kw in keywords:
#             if kw in text_lower:
#                 idx = text_lower.find(kw)
#                 section = text[idx:idx+1000]
#                 ingredients = self._parse_ingredients(section)
#                 if ingredients:
#                     break
        
#         if not ingredients:
#             common_ingredients = ["sugar", "salt", "oil", "flour", "water", "milk", "butter", "egg", "wheat"]
#             lines = text.split('\n')
#             for line in lines:
#                 if any(ci in line.lower() for ci in common_ingredients[:5]):
#                     ingredients.append(line.strip())
        
#         return ingredients[:15]
    
#     def _parse_ingredients(self, section):
#         """Parse ingredients from section text"""
#         import re
#         ingredients = []
        
#         parts = re.split(r'[,\n]', section[:800])
        
#         for part in parts:
#             part = part.strip()
#             if 3 < len(part) < 150:
#                 skip_words = ["ingredients", "contains", "may contain", "allergy", "warning"]
#                 if not any(skip in part.lower() for skip in skip_words):
#                     ingredients.append(part)
        
#         return ingredients[:12]
    
#     def get_llm_summary(self, text):
#         """Get LLM summary using Ollama"""
#         if not USE_LLM:
#             return "LLM summarization disabled."
        
#         if not text or len(text) < 20:
#             return "Not enough text for analysis."
        
#         try:
#             prompt = f"""You are a nutrition expert. Analyze this food label text and provide:

# 1. MAIN INGREDIENTS: List the key ingredients
# 2. ALLERGENS: Identify any common allergens (milk, eggs, nuts, wheat, soy, fish, shellfish)
# 3. NUTRITION HIGHLIGHTS: Key nutritional info (calories, sugar, fat, protein)
# 4. HEALTH NOTES: Any health concerns or positive aspects
# 5. SUMMARY: Brief summary of the product

# Text from food label:
# {text[:2000]}

# Provide a clear, concise response:
# """
            
#             response = requests.post(
#                 OLLAMA_URL,
#                 json={
#                     "model": OLLAMA_MODEL,
#                     "prompt": prompt,
#                     "stream": False,
#                     "options": {"temperature": 0.3, "num_predict": 500}
#                 },
#                 timeout=60
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 return result.get("response", "Could not generate summary.")
#             else:
#                 return f"LLM error: {response.status_code}"
                
#         except requests.exceptions.Timeout:
#             return "LLM request timed out. Try again."
#         except Exception as e:
#             logger.error(f"LLM error: {e}")
#             return f"LLM error: {str(e)}"
    
#     def create_pdf_report(self, text, ingredients, llm_summary, image_path):
#         """Create PDF report with OCR results and LLM analysis"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         pdf_path = f"/tmp/food_label_report_{timestamp}.pdf"
        
#         c = canvas.Canvas(pdf_path, pagesize=letter)
#         width, height = letter
#         y = height - 50
        
#         c.setFont("Helvetica-Bold", 18)
#         c.drawString(50, y, "Food Label Analysis Report")
#         y -= 40
        
#         c.setFont("Helvetica", 10)
#         c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         y -= 30
        
#         c.setFont("Helvetica-Bold", 14)
#         c.drawString(50, y, "Extracted Text:")
#         y -= 25
        
#         c.setFont("Helvetica", 10)
#         lines = simpleSplit(text[:1500], "Helvetica", 10, 500)
#         for line in lines[:25]:
#             if y < 50:
#                 c.showPage()
#                 y = height - 50
#             c.drawString(50, y, line)
#             y -= 15
        
#         if ingredients:
#             y -= 15
#             c.setFont("Helvetica-Bold", 14)
#             c.drawString(50, y, "Extracted Ingredients:")
#             y -= 25
            
#             c.setFont("Helvetica", 10)
#             for ing in ingredients[:15]:
#                 if y < 50:
#                     c.showPage()
#                     y = height - 50
#                 c.drawString(60, y, f"• {ing}")
#                 y -= 15
        
#         if llm_summary and "disabled" not in llm_summary and "error" not in llm_summary.lower():
#             y -= 15
#             c.setFont("Helvetica-Bold", 14)
#             c.drawString(50, y, "AI Analysis:")
#             y -= 25
            
#             c.setFont("Helvetica", 10)
#             lines = simpleSplit(llm_summary[:1500], "Helvetica", 10, 500)
#             for line in lines:
#                 if y < 50:
#                     c.showPage()
#                     y = height - 50
#                 c.drawString(60, y, line)
#                 y -= 15
        
#         c.setFont("Helvetica-Oblique", 8)
#         c.drawString(50, 30, "Generated by Food Label OCR Bot")
#         c.drawString(50, 20, f"Model: {self.model_name}")
        
#         c.save()
#         return pdf_path
    
#     def ocr(self, img_path, enhance=True):
#         start = time.time()
#         img = Image.open(img_path)
        
#         if enhance:
#             enhanced, _ = self.enhancer.enhance_image(img_path)
#             ocr_img = enhanced
#         else:
#             ocr_img = img.convert('L')
        
#         try:
#             text = pytesseract.image_to_string(ocr_img, lang=self.model_name)
#             used = self.model_name
#         except:
#             text = pytesseract.image_to_string(ocr_img, lang=self.fallback)
#             used = self.fallback
        
#         elapsed = time.time() - start
#         self.stats["total"] += 1
#         if text.strip():
#             self.stats["success"] += 1
#             self.stats["models"][used] = self.stats["models"].get(used, 0) + 1
#         else:
#             self.stats["failed"] += 1
        
#         ingredients = self.extract_ingredients(text)
#         llm_summary = self.get_llm_summary(text) if USE_LLM else "LLM summarization disabled."
        
#         return {
#             "text": text.strip(),
#             "time": elapsed,
#             "model": used,
#             "len": len(text.strip()),
#             "words": len(text.strip().split()),
#             "ingredients": ingredients,
#             "llm_summary": llm_summary
#         }
    
#     async def start(self, update, context):
#         status = "✅ Active" if MODEL_PATH.exists() else "⚠️ Fallback"
#         size = f"{MODEL_PATH.stat().st_size / 1024 / 1024:.1f}MB" if MODEL_PATH.exists() else "N/A"
#         llm_status = "✅ Connected" if self._check_ollama() else "❌ Not connected"
        
#         msg = f"""
# <b>🤖 Food Label OCR Bot with AI Analysis</b>

# <b>Model:</b> {status} {size}
# <b>Enhancement:</b> {"ON" if ENHANCE_IMAGE else "OFF"}
# <b>LLM (Mistral):</b> {llm_status}

# <b>Features:</b>
# • Extract ingredients from food labels
# • 119% better contrast enhancement
# • 18.6% better character accuracy
# • AI-powered ingredient analysis
# • PDF report generation

# <b>Commands:</b>
# /start - Welcome
# /help - Help guide
# /stats - Statistics
# /model - Model info
# /compare - Compare with base model
# /enhance - Toggle enhancement
# /pdf - Generate PDF report (after scanning)

# Send a food label photo to start!
# """
#         await update.message.reply_text(msg, parse_mode='HTML')
    
#     async def help(self, update, context):
#         msg = """
# <b>📚 Help Guide</b>

# 1. Take a photo of a food label
# 2. Send it to this bot
# 3. Get extracted text and ingredients
# 4. Use /pdf to download PDF report
# 5. Get AI analysis of ingredients

# <b>Commands:</b>
# /start - Welcome
# /help - This guide
# /stats - Usage statistics
# /model - Model performance
# /compare - Compare with base OCR
# /enhance - Toggle enhancement
# /pdf - Download PDF report (after scanning)

# <b>Tips:</b>
# • Good lighting = better results
# • Hold camera steady
# • Supported: JPG, PNG, TIFF
# """
#         await update.message.reply_text(msg, parse_mode='HTML')
    
#     async def stats(self, update, context):
#         t = self.stats["total"]
#         s = self.stats["success"]
#         f = self.stats["failed"]
#         rate = (s/t*100) if t else 0
        
#         msg = f"""
# <b>📊 OCR Statistics</b>

# <b>Total requests:</b> {t}
# <b>Successful:</b> {s} ({rate:.1f}%)
# <b>Failed:</b> {f}

# <b>Models Used:</b>
# """
#         for m, c in self.stats["models"].items():
#             msg += f"• {m}: {c}\n"
        
#         await update.message.reply_text(msg, parse_mode='HTML')
    
#     async def model_info(self, update, context):
#         if MODEL_PATH.exists():
#             mb = MODEL_PATH.stat().st_size / 1024 / 1024
#             mod = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
#             msg = f"""
# <b>🔬 Fine-tuned Model</b>

# <b>Name:</b> {self.model_name}
# <b>Size:</b> {mb:.2f} MB
# <b>Created:</b> {mod.strftime('%Y-%m-%d %H:%M')}

# <b>Performance:</b>
# • CER: 48.1% (18.6% better than base)
# • WER: 69.2% (28.2% better than base)
# • Contrast Enhancement: +119%
# • Training Samples: 5,000
# """
#         else:
#             msg = f"⚠️ Model not found at: {MODEL_PATH}"
#         await update.message.reply_text(msg, parse_mode='HTML')
    
#     async def compare(self, update, context):
#         await update.message.reply_text(
#             "📸 <b>Model Comparison Mode</b>\n\nSend me an image and I'll show results from:\n"
#             "1. Base Tesseract model (eng)\n"
#             "2. Your fine-tuned model (eng_textocr)\n\n"
#             "This will show the improvement from your training!",
#             parse_mode='HTML'
#         )
#         context.user_data['compare'] = True
    
#     async def enhance(self, update, context):
#         current = context.user_data.get('enhance', ENHANCE_IMAGE)
#         context.user_data['enhance'] = not current
#         status = "ON ✅" if context.user_data['enhance'] else "OFF ❌"
#         await update.message.reply_text(f"🖼️ <b>Image Enhancement:</b> {status}\n\n119% better contrast when ON", parse_mode='HTML')
    
#     async def pdf_report(self, update, context):
#         if not context.user_data.get('last_result'):
#             await update.message.reply_text("❌ No scan data found. Please send a food label image first.")
#             return
        
#         result = context.user_data['last_result']
        
#         await update.message.reply_text("📄 Generating PDF report...")
        
#         pdf_path = self.create_pdf_report(
#             result['text'],
#             result['ingredients'],
#             result['llm_summary'],
#             None
#         )
        
#         with open(pdf_path, 'rb') as f:
#             await update.message.reply_document(
#                 document=f,
#                 filename=f"food_label_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
#                 caption="📋 Food Label Analysis Report"
#             )
        
#         os.remove(pdf_path)
    
#     async def handle_photo(self, update, context):
#         user = update.effective_user
#         status_msg = await update.message.reply_text("🔄 Processing image...")
        
#         try:
#             photo = update.message.photo[-1]
#             file = await photo.get_file()
#             path = f"/tmp/ocr_{user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#             await file.download_to_drive(path)
            
#             compare_mode = context.user_data.get('compare', False)
#             use_enhance = context.user_data.get('enhance', ENHANCE_IMAGE)
            
#             if compare_mode:
#                 await status_msg.edit_text("🔄 Comparing models...")
                
#                 img = Image.open(path).convert('L')
#                 base_start = time.time()
#                 base_text = pytesseract.image_to_string(img, lang='eng').strip()
#                 base_time = time.time() - base_start
                
#                 ft_result = self.ocr(path, use_enhance)
                
#                 imp = len(ft_result['text']) - len(base_text)
#                 imp_pct = (imp / len(base_text) * 100) if len(base_text) > 0 else 0
                
#                 msg = f"""
# <b>🔍 OCR Model Comparison</b>

# <b>Base Model (eng):</b>
# Time: {base_time:.2f}s | Length: {len(base_text)} chars

# <b>Fine-tuned Model ({self.model_name}):</b>
# Time: {ft_result['time']:.2f}s | Length: {len(ft_result['text'])} chars

# <b>📈 Improvement:</b> +{imp} chars ({imp_pct:+.1f}%)
# """
#                 await status_msg.edit_text(msg, parse_mode='HTML')
#                 context.user_data['compare'] = False
#             else:
#                 result = self.ocr(path, use_enhance)
#                 context.user_data['last_result'] = result
                
#                 # Build message with HTML (no Markdown issues)
#                 msg = "<b>📸 Food Label Analysis</b>\n\n"
                
#                 # Add extracted text
#                 msg += "<b>Extracted Text:</b>\n"
#                 escaped_text = escape_html(result['text'][:500])
#                 msg += f"<code>{escaped_text}</code>\n\n"
                
#                 # Add ingredients
#                 msg += "<b>Detected Ingredients:</b>\n"
#                 if result['ingredients']:
#                     for ing in result['ingredients'][:10]:
#                         escaped_ing = escape_html(ing)
#                         msg += f"• {escaped_ing}\n"
#                 else:
#                     msg += "No ingredients detected\n"
#                 msg += "\n"
                
#                 # Add AI analysis
#                 if result['llm_summary'] and "disabled" not in result['llm_summary'] and "error" not in result['llm_summary'].lower():
#                     msg += "<b>🤖 AI Analysis:</b>\n"
#                     escaped_summary = escape_html(result['llm_summary'][:500])
#                     msg += f"{escaped_summary}\n\n"
                
#                 # Add statistics
#                 msg += f"""<b>Statistics:</b>
# • Model: {result['model']}
# • Time: {result['time']:.2f}s
# • Characters: {result['len']}
# • Words: {result['words']}

# 📄 Use /pdf to download full report
# """
                
#                 await status_msg.edit_text(msg, parse_mode='HTML')
            
#             os.remove(path)
            
#         except Exception as e:
#             logger.error(f"Error: {e}")
#             await status_msg.edit_text(f"❌ Error: {str(e)}")
    
#     async def handle_document(self, update, context):
#         user = update.effective_user
#         status_msg = await update.message.reply_text("🔄 Processing document...")
        
#         try:
#             file = await update.message.document.get_file()
#             filename = update.message.document.file_name or "image.jpg"
            
#             if not any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
#                 await status_msg.edit_text("❌ Please send an image file (JPG, PNG, TIFF)")
#                 return
            
#             path = f"/tmp/ocr_{user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#             await file.download_to_drive(path)
            
#             result = self.ocr(path, ENHANCE_IMAGE)
#             context.user_data['last_result'] = result
            
#             msg = "<b>📄 Document OCR Result</b>\n\n"
#             msg += "<b>Extracted Text:</b>\n"
#             escaped_text = escape_html(result['text'][:500])
#             msg += f"<code>{escaped_text}</code>\n\n"
            
#             msg += "<b>Detected Ingredients:</b>\n"
#             if result['ingredients']:
#                 for ing in result['ingredients'][:10]:
#                     escaped_ing = escape_html(ing)
#                     msg += f"• {escaped_ing}\n"
#             else:
#                 msg += "No ingredients detected\n"
#             msg += "\n"
            
#             msg += f"""<b>Statistics:</b>
# • Model: {result['model']}
# • Time: {result['time']:.2f}s
# • Characters: {result['len']}

# 📄 Use /pdf to download full report
# """
#             await status_msg.edit_text(msg, parse_mode='HTML')
#             os.remove(path)
            
#         except Exception as e:
#             logger.error(f"Error: {e}")
#             await status_msg.edit_text(f"❌ Error: {str(e)}")
    
#     def run(self):
#         if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
#             print("\n" + "="*60)
#             print("❌ NO BOT TOKEN CONFIGURED")
#             print("="*60)
#             return
        
#         app = Application.builder().token(BOT_TOKEN).build()
        
#         app.add_handler(CommandHandler("start", self.start))
#         app.add_handler(CommandHandler("help", self.help))
#         app.add_handler(CommandHandler("stats", self.stats))
#         app.add_handler(CommandHandler("model", self.model_info))
#         app.add_handler(CommandHandler("compare", self.compare))
#         app.add_handler(CommandHandler("enhance", self.enhance))
#         app.add_handler(CommandHandler("pdf", self.pdf_report))
#         app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
#         app.add_handler(MessageHandler(filters.Document.IMAGE, self.handle_document))
        
#         print("\n" + "="*60)
#         print("🤖 FOOD LABEL OCR BOT STARTED")
#         print("="*60)
#         print(f"Model: {self.model_name}")
#         print(f"LLM (Mistral): {'Enabled' if USE_LLM else 'Disabled'}")
#         print(f"PDF Reports: Enabled")
#         print("="*60)
#         print("\nBot is running! Send food label images to test")
#         print("Commands: /start, /help, /stats, /model, /compare, /enhance, /pdf")
#         print("Press Ctrl+C to stop\n")
        
#         app.run_polling()


# if __name__ == "__main__":
#     bot = FoodLabelOCRBot()
#     bot.run()
    
    
    
    # telegram_bot/bot.py

# telegram_bot/bot.py
"""
Food Label OCR Bot with AI Analysis - Complete Working Version
Copy and paste this entire code into bot.py
"""

import os
import sys
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

sys.path.append(str(Path(__file__).parent.parent))

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
)
from PIL import Image
import pytesseract

from ocr.image_enhanced import ImageEnhancer

# ============================================================
# CONFIGURATION - CHANGE YOUR TOKEN HERE
# ============================================================

BOT_TOKEN = "8418376380:AAG_NBpcr87ulQ6ESxM0LyR1KOAM8Zrf6Gk"

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "tesseract" / "eng_textocr.traineddata"

TESSERACT_LANG = "eng_textocr"
FALLBACK_LANG = "eng"

ENHANCE_IMAGE = True
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

USE_LLM = True
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def escape_html(text):
    """Escape special characters for HTML"""
    if not text:
        return ""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    return text


class FoodLabelBot:
    """Food Label OCR Bot with AI Analysis"""
    
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.model_name = TESSERACT_LANG
        self.fallback = FALLBACK_LANG
        self.stats = {"total": 0, "success": 0, "failed": 0, "models": {}}
        self.user_data = {}
        self._check_model()
    
    def _check_model(self):
        if MODEL_PATH.exists():
            mb = MODEL_PATH.stat().st_size / 1024 / 1024
            logger.info(f"✅ Model found: {MODEL_PATH} ({mb:.1f}MB)")
            return True
        logger.warning(f"⚠️ Model not found, using fallback")
        return False
    
    def get_llm_analysis(self, text, analysis_type="full"):
        """Get LLM analysis using Mistral"""
        if not text or len(text) < 20:
            return "Not enough text for analysis. Please send a clearer photo."
        
        prompts = {
            "ingredients": "List only the main ingredients from this food label. Format as bullet points.",
            "allergens": "List any allergens found (milk, eggs, nuts, wheat, soy, fish, shellfish). If none, say 'No common allergens detected'.",
            "nutrition": "Extract nutritional information: calories, sugar, fat, protein, sodium if available.",
            "health": "Provide health notes: good aspects and concerns about this product.",
            "full": """Analyze this food label and provide:
1. MAIN INGREDIENTS
2. ALLERGENS
3. NUTRITION HIGHLIGHTS
4. HEALTH NOTES
5. SUMMARY"""
        }
        
        prompt = prompts.get(analysis_type, prompts["full"])
        full_prompt = f"{prompt}\n\nText: {text[:2000]}"
        
        try:
            logger.info(f"Sending to Ollama: {analysis_type}")
            
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 500}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Could not generate summary.")
            else:
                return f"LLM error: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Start with: ollama serve"
        except requests.exceptions.Timeout:
            return "⏰ LLM request timed out. Try again."
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"❌ Error: {str(e)}"
    
    def ocr_on_image(self, image_path, enhance=True):
        """Run OCR on image"""
        start = time.time()
        
        if enhance:
            enhanced, _ = self.enhancer.enhance_image(image_path)
            ocr_img = enhanced
        else:
            ocr_img = Image.open(image_path).convert('L')
        
        try:
            text = pytesseract.image_to_string(ocr_img, lang=self.model_name)
            used = self.model_name
        except:
            text = pytesseract.image_to_string(ocr_img, lang=self.fallback)
            used = self.fallback
        
        elapsed = time.time() - start
        self.stats["total"] += 1
        if text.strip():
            self.stats["success"] += 1
            self.stats["models"][used] = self.stats["models"].get(used, 0) + 1
        else:
            self.stats["failed"] += 1
        
        return text.strip(), used, elapsed
    
    def create_pdf_report(self, text, analysis):
        """Create PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"/tmp/food_report_{timestamp}.pdf"
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        y = height - 50
        
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, y, "Food Label Analysis Report")
        y -= 40
        
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 30
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Extracted Text:")
        y -= 25
        
        c.setFont("Helvetica", 10)
        lines = simpleSplit(text[:1500], "Helvetica", 10, 500)
        for line in lines[:25]:
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(50, y, line)
            y -= 15
        
        if analysis and "Error" not in analysis:
            y -= 15
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y, "AI Analysis:")
            y -= 25
            
            c.setFont("Helvetica", 10)
            lines = simpleSplit(analysis[:1500], "Helvetica", 10, 500)
            for line in lines:
                if y < 50:
                    c.showPage()
                    y = height - 50
                c.drawString(60, y, line)
                y -= 15
        
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(50, 30, "Generated by Food Label OCR Bot")
        c.drawString(50, 20, f"Model: {self.model_name}")
        
        c.save()
        return pdf_path
    
    async def start(self, update, context):
        msg = """
<b>🤖 Food Label OCR Bot</b>

<b>How to use:</b>
1. Send a photo of a food label
2. Click one of the buttons:
   • 🥫 Ingredients
   • ⚠️ Allergens
   • 🍎 Nutrition
   • 💚 Health Notes
   • 📋 Full Analysis
3. Get AI-powered analysis!

<b>Commands:</b>
/start - Welcome
/help - Help guide
/stats - Statistics
/model - Model info
/pdf - Download PDF report

<b>Model:</b> Fine-tuned OCR (18.6% better accuracy)
"""
        await update.message.reply_text(msg, parse_mode='HTML')
    
    async def help(self, update, context):
        msg = """
<b>📚 Help Guide</b>

<b>Step 1:</b> Take a clear photo of a food label
<b>Step 2:</b> Send it to this bot
<b>Step 3:</b> Choose what you want to know
<b>Step 4:</b> Get AI analysis!

<b>Features:</b>
• Extracts text from food labels
• AI analysis with Mistral
• PDF report generation
• 119% contrast enhancement
• 18.6% better character accuracy

<b>Commands:</b>
/start - Welcome message
/help - This guide
/stats - Usage statistics
/model - Model information
/pdf - Download PDF report
"""
        await update.message.reply_text(msg, parse_mode='HTML')
    
    async def stats(self, update, context):
        t = self.stats["total"]
        s = self.stats["success"]
        f = self.stats["failed"]
        rate = (s/t*100) if t else 0
        
        msg = f"""
<b>📊 OCR Statistics</b>

<b>Total requests:</b> {t}
<b>Successful OCR:</b> {s} ({rate:.1f}%)
<b>Failed OCR:</b> {f}

<b>Models Used:</b>
"""
        for m, c in self.stats["models"].items():
            msg += f"• {m}: {c}\n"
        
        await update.message.reply_text(msg, parse_mode='HTML')
    
    async def model_info(self, update, context):
        if MODEL_PATH.exists():
            mb = MODEL_PATH.stat().st_size / 1024 / 1024
            mod = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
            msg = f"""
<b>🔬 Fine-tuned Model Information</b>

<b>Name:</b> {self.model_name}
<b>Size:</b> {mb:.2f} MB
<b>Created:</b> {mod.strftime('%Y-%m-%d %H:%M')}

<b>Performance Metrics:</b>
• CER: 48.1% (18.6% better than base)
• WER: 69.2% (28.2% better than base)
• Contrast Enhancement: +119%
• Training Samples: 5,000

<b>Usage:</b> Images are automatically processed with this model
"""
        else:
            msg = f"⚠️ Model not found at: {MODEL_PATH}\n\nPlease train the model first."
        
        await update.message.reply_text(msg, parse_mode='HTML')
    
    async def pdf_command(self, update, context):
        user_id = update.effective_user.id
        
        if user_id not in self.user_data:
            await update.message.reply_text("❌ No data found. Please send a food label photo first.")
            return
        
        text = self.user_data.get(user_id, {}).get('text', '')
        analysis = self.user_data.get(user_id, {}).get('analysis', 'No analysis available')
        
        if not text:
            await update.message.reply_text("❌ No text extracted. Please send a photo first.")
            return
        
        await update.message.reply_text("📄 Generating PDF report...")
        
        try:
            pdf_path = self.create_pdf_report(text, analysis)
            
            with open(pdf_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=f"food_label_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    caption="📋 Food Label Analysis Report"
                )
            
            os.remove(pdf_path)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error generating PDF: {str(e)}")
    
    async def handle_photo(self, update, context):
        user_id = update.effective_user.id
        status_msg = await update.message.reply_text("🔄 Processing image...")
        
        try:
            photo = update.message.photo[-1]
            file = await photo.get_file()
            path = f"/tmp/ocr_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            await file.download_to_drive(path)
            
            text, used, elapsed = self.ocr_on_image(path, enhance=True)
            
            # Store data
            if user_id not in self.user_data:
                self.user_data[user_id] = {}
            self.user_data[user_id]['text'] = text
            self.user_data[user_id]['image_path'] = path
            self.user_data[user_id]['model'] = used
            
            preview = text[:200] if text else "No text detected"
            
            # Show results and analysis options
            keyboard = [
                [InlineKeyboardButton("🥫 Ingredients", callback_data="ask_ingredients")],
                [InlineKeyboardButton("⚠️ Allergens", callback_data="ask_allergens")],
                [InlineKeyboardButton("🍎 Nutrition", callback_data="ask_nutrition")],
                [InlineKeyboardButton("💚 Health Notes", callback_data="ask_health")],
                [InlineKeyboardButton("📋 Full Analysis", callback_data="ask_full")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await status_msg.edit_text(
                f"<b>✅ Image Processed Successfully!</b>\n\n"
                f"<b>Model:</b> {used}\n"
                f"<b>Time:</b> {elapsed:.2f}s\n"
                f"<b>Characters:</b> {len(text)}\n\n"
                f"<b>Preview:</b>\n<code>{escape_html(preview)}</code>\n\n"
                f"<b>What would you like to know?</b>",
                reply_markup=reply_markup,
                parse_mode='HTML'
            )
            
        except Exception as e:
            logger.error(f"Error: {e}")
            await status_msg.edit_text(f"❌ Error: {str(e)}")
    
    async def analysis_callback(self, update, context):
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        analysis_type = query.data.split("_")[1]
        
        if user_id not in self.user_data or 'text' not in self.user_data[user_id]:
            await query.edit_message_text("❌ Please send a photo first.")
            return
        
        text = self.user_data[user_id]['text']
        
        await query.edit_message_text(f"🤖 Analyzing {analysis_type}... Please wait.")
        
        # Get analysis
        analysis = self.get_llm_analysis(text, analysis_type)
        
        # Store analysis for PDF
        self.user_data[user_id]['analysis'] = analysis
        
        # Prepare response
        type_names = {
            "ingredients": "🥫 INGREDIENTS",
            "allergens": "⚠️ ALLERGENS",
            "nutrition": "🍎 NUTRITION FACTS",
            "health": "💚 HEALTH NOTES",
            "full": "📋 FULL ANALYSIS"
        }
        
        msg = f"""
<b>{type_names.get(analysis_type, 'ANALYSIS')}</b>

<code>{escape_html(analysis[:1200])}</code>

<i>Use /pdf to download full report with both text and analysis</i>
"""
        
        await query.edit_message_text(msg, parse_mode='HTML')
    
    async def handle_document(self, update, context):
        user_id = update.effective_user.id
        status_msg = await update.message.reply_text("🔄 Processing document...")
        
        try:
            file = await update.message.document.get_file()
            filename = update.message.document.file_name or "image.jpg"
            
            if not any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
                await status_msg.edit_text("❌ Please send an image file (JPG, PNG, TIFF)")
                return
            
            path = f"/tmp/ocr_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            await file.download_to_drive(path)
            
            text, used, elapsed = self.ocr_on_image(path, enhance=True)
            
            if user_id not in self.user_data:
                self.user_data[user_id] = {}
            self.user_data[user_id]['text'] = text
            
            preview = text[:200] if text else "No text detected"
            
            keyboard = [
                [InlineKeyboardButton("🥫 Ingredients", callback_data="ask_ingredients")],
                [InlineKeyboardButton("⚠️ Allergens", callback_data="ask_allergens")],
                [InlineKeyboardButton("🍎 Nutrition", callback_data="ask_nutrition")],
                [InlineKeyboardButton("📋 Full Analysis", callback_data="ask_full")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await status_msg.edit_text(
                f"<b>✅ Document Processed</b>\n\n"
                f"<b>Model:</b> {used}\n"
                f"<b>Time:</b> {elapsed:.2f}s\n"
                f"<b>Characters:</b> {len(text)}\n\n"
                f"<b>Preview:</b>\n<code>{escape_html(preview)}</code>\n\n"
                f"<b>Choose analysis:</b>",
                reply_markup=reply_markup,
                parse_mode='HTML'
            )
            
            os.remove(path)
            
        except Exception as e:
            logger.error(f"Error: {e}")
            await status_msg.edit_text(f"❌ Error: {str(e)}")
    
    def run(self):
        if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
            print("\n" + "="*60)
            print("❌ NO BOT TOKEN CONFIGURED")
            print("="*60)
            print("\nGet token from @BotFather on Telegram:")
            print("1. Search @BotFather")
            print("2. Send /newbot")
            print("3. Choose name: Food Label OCR Bot")
            print("4. Choose username: food_label_ocr_bot")
            print("5. Copy the token and paste in BOT_TOKEN variable")
            print("="*60)
            return
        
        # Stop any existing bot instances
        try:
            app = Application.builder().token(BOT_TOKEN).build()
            
            app.add_handler(CommandHandler("start", self.start))
            app.add_handler(CommandHandler("help", self.help))
            app.add_handler(CommandHandler("stats", self.stats))
            app.add_handler(CommandHandler("model", self.model_info))
            app.add_handler(CommandHandler("pdf", self.pdf_command))
            app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
            app.add_handler(MessageHandler(filters.Document.IMAGE, self.handle_document))
            app.add_handler(CallbackQueryHandler(self.analysis_callback, pattern="^ask_"))
            
            print("\n" + "="*60)
            print("🤖 FOOD LABEL OCR BOT STARTED")
            print("="*60)
            print(f"Model: {self.model_name}")
            print(f"Model exists: {MODEL_PATH.exists()}")
            print(f"LLM: {OLLAMA_MODEL} (via Ollama)")
            print(f"PDF Reports: Enabled")
            print("="*60)
            print("\nBot is running! Send food label images to test")
            print("Commands: /start, /help, /stats, /model, /pdf")
            print("Press Ctrl+C to stop\n")
            
            app.run_polling()
            
        except Exception as e:
            if "Conflict" in str(e):
                print("\n⚠️ Another bot instance is running!")
                print("Stop it with: pkill -f 'bot.py'")
                print("Then run this script again")
            else:
                print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    bot = FoodLabelBot()
    bot.run()
