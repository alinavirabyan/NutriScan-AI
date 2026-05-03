<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Telegram-Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white"/>
<img src="https://img.shields.io/badge/Tesseract-OCR-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Ollama-Mistral_7B-7C3AED?style=for-the-badge"/>


# 🍎 NutriScan AI

### AI-Powered Food Label Analysis Bot

*Diploma Project — Fine-tuning OCR for food label recognition in noisy, real-world conditions*

[Features](#-features) · [Model Performance](#-model-performance) · [Quick Start](#-quick-start) · [Project Structure](#-project-structure) · [Training Pipeline](#-training-pipeline)

</div>

---

## 📋 Overview

**NutriScan AI** is an intelligent Telegram bot that analyzes food product labels in real time using a fine-tuned Tesseract OCR model combined with Ollama's Mistral 7B language model.

Users simply send a photo of any food label. The system extracts text from the image, identifies ingredients and allergens, evaluates nutritional quality, and delivers a structured health report — all within seconds.

> 🎓 This project was developed as a diploma thesis focusing on the fine-tuning of Tesseract OCR for robust food label recognition under noisy, real-world imaging conditions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Smart OCR** | Fine-tuned `eng_textocr` model optimized for food label typography and noisy conditions |
| 🧠 **AI Analysis** | Deep ingredient and nutrition analysis powered by Mistral 7B via Ollama |
| 🚨 **Allergen Detection** | Detects 8 common allergens: milk, gluten, peanuts, soy, eggs, lactose, nuts, wheat |
| 📊 **Health Rating** | Clear three-tier rating system: 🟢 Healthy · 🟡 Moderate · 🔴 Unhealthy |
| 📄 **PDF Reports** | Auto-generated structured analysis reports ready to download |
| 💬 **AI Q&A** | Ask follow-up nutrition questions about any scanned product |
| 👤 **User Profiles** | Personalized analysis based on age, weight, and personal allergen list |
| 📜 **Scan History** | Browse and track all previously saved scans |
| ℹ️ **Allergen Guide** | In-bot educational descriptions for every supported allergen |

---

## 🎯 Model Performance

### 🧪 Training Details

| Parameter | Value |
|---|---|
| Dataset size | 273,750 word crops |
| Training iterations | 1,000,000 |
| Total training time | ~9.3 hours |
| Base model | Tesseract `eng` |
| Fine-tuned model | `eng_textocr` |

| Metric | At Start | Best Achieved |
|---|---|---|
| BCER (↓ lower is better) | 85.0% | **15.5%** |
| BWER (↓ lower is better) | 84.0% | — |

---

### 📊 Evaluation Results — 68,438 Samples · 240 min

| Metric | Base Tesseract | Fine-tuned Model | Improvement |
|---|---|---|---|
| CER ↓ | 72.8% | **42.2%** | −42.1% |
| WER ↓ | 124.5% | **71.8%** | −42.3% |
| Accuracy ↑ | 11.0% | **34.2%** | +23.1% |
| F1 Score ↑ | 16.5% | **34.4%** | +18.0% |
| Exact Match ↑ | 8.0% | **33.7%** | +25.7% |

> The fine-tuned model achieves **3.1× better accuracy**, **2.1× better F1 score**, and **4.2× better exact match rate** compared to the base Tesseract engine — evaluated on 68,438 real-world food label samples.

---

## 🚀 Quick Start

### 🔧 Prerequisites

Before running the bot, make sure you have the following installed:

- **Python** 3.10 or higher
- **Tesseract OCR** 5.0 or higher
- **Ollama** with the Mistral model pulled
- A valid **Telegram Bot Token** (obtain from [@BotFather](https://t.me/BotFather))

---

### ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/alinavirabyan/DiplomaWork.git
cd Final_Work
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Install the custom OCR model**
```bash
sudo cp models/tesseract/eng_textocr.traineddata \
/usr/share/tesseract-ocr/5/tessdata/
```

**4. Start Ollama and pull the Mistral model**
```bash
ollama serve &
ollama pull mistral
```

**5. Configure environment variables**

Create a `.env` file in the project root:
```env
TELEGRAM_BOT_TOKEN=your_token_here
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=mistral
TESSERACT_MODEL_DIR=/usr/share/tesseract-ocr/5/tessdata
```

**6. Run the bot**
```bash
python scripts/bot.py
```

---

## 📁 Project Structure

```
Final_Work/
│
├── scripts/
│ ├── bot.py # Telegram bot — main entry point
│ ├── preprocess.py # Dataset preprocessing pipeline
│ ├── train-model.py # Tesseract fine-tuning script
│ └── evaluate_model.py # Model evaluation & metrics
│
├── models/
│ └── tesseract/
│ └── eng_textocr.traineddata # Fine-tuned OCR model
│
├── custom/
│ └── charts/ # Training & evaluation charts
│
├── requirements.txt
└── README.md
```

---

## 🏋️ Training Pipeline

The model is trained in three sequential steps:

**Step 1 — Preprocess the dataset**
```bash
python scripts/preprocess.py
```
Cleans and prepares raw label images into word-crop training samples.

**Step 2 — Fine-tune the OCR model**
```bash
python scripts/train-model.py
```
Runs 1,000,000 training iterations of Tesseract LSTM fine-tuning on the prepared dataset.

**Step 3 — Evaluate the model**
```bash
python scripts/evaluate_model.py
```
Computes CER, WER, Accuracy, and F1 Score against the held-out test set and generates comparison charts.

---

## 🤖 How the Bot Works

```
User sends photo
↓
Image enhancement & preprocessing (OpenCV)
↓
OCR text extraction (fine-tuned eng_textocr + EasyOCR fallback)
↓
LLM analysis via Ollama / Mistral 7B
↓
Allergen matching against user profile
↓
Health rating + summary delivered to user
↓
Optional: PDF report · AI Q&A · Save to history
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Bot framework | python-telegram-bot 20+ |
| OCR engine | Tesseract 5 (fine-tuned) + EasyOCR |
| Image processing | OpenCV, Pillow |
| Language model | Ollama — Mistral 7B |
| PDF generation | ReportLab |
| Data storage | JSON flat-file database |
| Language | Python 3.10+ |

---

## 🎥 Demo

<p align="center">
<img src="outputs/evaluation_results/demo/animation.gif" width="500"/>
</p>

---

<div align="center">

Made with ❤️ as a Diploma Project

**NutriScan AI** — *Scan smarter. Eat better.*

</div>

