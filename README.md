# OCR Bot

A Telegram bot that extracts UK postal addresses from images. Send a photo of a letter, envelope, or document and get back a clean, formatted address.

## What It Does

1. Receive a photo via Telegram
2. Run OCR on the image using a remote EasyOCR API
3. Pass the extracted text to an LLM to identify and clean up the postal address
4. Return the formatted UK address to the user

## Tech Stack

- **Bot**: python-telegram-bot
- **OCR**: EasyOCR via Hugging Face Spaces
- **LLM**: Hugging Face Inference API for address extraction
- **Image Processing**: OpenCV, Pillow

## Setup

```bash
git clone https://github.com/sedici16/ocr_bot.git
cd ocr_bot
pip install -r requirements.txt
```

Create a `.env` file:

```
HF_TOKEN=your-huggingface-token
TELEGRAM_BOT_TOKEN=your-bot-token
```

Run:

```bash
python final.py
```

## How It Works

The bot uses a two-stage pipeline:

1. **OCR Stage** - The image is sent to a hosted EasyOCR model that extracts raw text
2. **LLM Stage** - The raw text is passed to a language model with a prompt to extract and correct UK-style addresses (fixing common OCR errors like O/0 confusion)
