import logging
import os
import tempfile
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from telegram.ext import Updater, MessageHandler, Filters
from telegram import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.utils.helpers import escape_markdown
from huggingface_hub import InferenceClient
from gradio_client import Client, handle_file

# Load Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

# Connect to Hugging Face OCR API
ocr_client = Client("theoracle/easyocr-api", hf_token=HF_TOKEN)

# Connect to Hugging Face chat completion API (LLM)
client = InferenceClient(provider="novita", api_key=HF_TOKEN)

# Enable logging
logging.basicConfig(level=logging.INFO)

def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Optional: Convert a PIL image to black-and-white using adaptive thresholding.
    This can sometimes improve OCR results.
    """
    img = np.array(pil_image.convert("L"))  # Convert to grayscale
    bw = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2
    )
    return Image.fromarray(bw)

def remote_ocr(image: Image.Image) -> str:
    """
    Send an image to the Hugging Face OCR endpoint and return the extracted text.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        image.save(temp.name)
        temp.flush()

        result = ocr_client.predict(
            image=handle_file(temp.name),
            api_name="/predict"
        )
        return result

def extract_address(input_text: str) -> str:
    """
    Use a large language model (LLM) to extract a UK-style postal address from OCR text.
    """
    prompt = f"""
    Please extract only the physical address from the OCR text. The addresses are UK ones, like:
    24 Church Street, Birmingham, B3 2RT
    11 Castle Boulevard, Nottingham, NG7 1FT
    3 Clarendon Park Road, Leicester, LE2 3AJ
    6 Abbey Street, Nuneaton, CV11 5BT

    ⚠️ Correct OCR errors like the letter O instead of zero (0).
    Return only the address, no names, phone numbers, or comments.

    USER INPUT:
    {input_text}

    address output
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

def handle_photo(update: Update, context: CallbackContext):
    """
    Handle incoming photo from Telegram user.
    Runs OCR and address extraction, and replies with results and Google Maps link.
    """
    # Get the largest photo version and download it
    photo_file = update.message.photo[-1].get_file()
    image_bytes = photo_file.download_as_bytearray()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    update.message.reply_text("🧠 Processing image...")

    try:
        # Step 1: OCR
        text = remote_ocr(image)
        if text.strip():
            # Step 2: Display raw OCR text
            safe_text = escape_markdown(text, version=2)
            update.message.reply_text(f"📄 *OCR Result:*\n\n{safe_text}", parse_mode='MarkdownV2')

            # Step 3: Extract address from OCR text using LLM
            address = extract_address(text)
            safe_address = escape_markdown(address, version=2)
            update.message.reply_text(f"📍 *Detected Address:*\n\n{safe_address}", parse_mode='MarkdownV2')

            # Step 4: Google Maps link
            encoded_address = address.replace(" ", "+")
            maps_url = f"https://www.google.com/maps/search/{encoded_address}"
            safe_url = escape_markdown(maps_url, version=2)
            update.message.reply_text(f"🗺️ [View on Google Maps]({safe_url})", parse_mode='MarkdownV2')

        else:
            update.message.reply_text("❗No readable text detected.")
    except Exception as e:
        update.message.reply_text(f"⚠️ OCR error: {e}")

def main():
    """
    Main entry point for the bot. Loads the token, starts polling for messages.
    """
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN environment variable not set.")
        return

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Respond to any photo sent to the bot
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    print("🤖 Bot is running...")
    updater.idle()

if __name__ == '__main__':
    main()
