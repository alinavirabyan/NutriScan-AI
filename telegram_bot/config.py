# Create telegram_bot/config.py
cat > telegram_bot/config.py << 'EOF'
# telegram_bot/config.py
"""
Telegram Bot Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Bot token from @BotFather
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")

# Model paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "tesseract" / "eng_textocr.traineddata"

# OCR settings
TESSERACT_LANG = "eng_textocr"  # Your fine-tuned model
FALLBACK_LANG = "eng"            # Fallback to base model

# Image enhancement settings
ENHANCE_IMAGE = True  # Apply enhancement before OCR

# Mistral LLM settings (optional)
USE_SUMMARIZATION = True
MISTRAL_API_URL = os.getenv("MISTRAL_API_URL", "http://localhost:11434/api/generate")

# Bot settings
MAX_IMAGE_SIZE_MB = 20
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
EOF