import os
import json
import tempfile
import urllib.parse
from typing import List, Optional
from PIL import Image
import gradio as gr
from gtts import gTTS
import whisper
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. API Configuration & Model Initialization
# ---------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

print("Loading Whisper ASR model ('tiny')...")
whisper_model = whisper.load_model("tiny", device="cpu")


# ---------------------------------------------------------
# 2. Pydantic Structured Output Schema
# ---------------------------------------------------------
class ProductMetadata(BaseModel):
    product_name: str = Field(description="Specific brand and product title identified from the image or query")
    category: str = Field(description="Category, e.g., Footwear, Electronics, Apparel, Personal Care, Grocery")
    estimated_price: str = Field(description="Estimated price or price range in INR (₹)")
    key_features: List[str] = Field(description="Key attributes, color, silhouette, or technical specifications")
    recommendation_summary: str = Field(description="Styling advice, matching accessories, or receipt breakdown")


# ---------------------------------------------------------
# 3. Dynamic Marketplace Link Generator
# ---------------------------------------------------------
def generate_store_cards(item_name: str) -> str:
    safe_query = urllib.parse.quote_plus(item_name if item_name else "product")
    
    amazon_url = f"https://www.amazon.in/s?k={safe_query}"
    flipkart_url = f"https://www.flipkart.com/search?q={safe_query}"
    google_shop_url = f"https://www.google.com/search?tbm=shop&q={safe_query}"
    myntra_url = f"https://www.myntra.com/{safe_query}"

    return f"""
---
### 🛍️ Direct Store & Purchase Links
Click below to view real-time listings, compare prices, and check availability:

* 🛒 **[Open in Amazon India]({amazon_url})** — Search Prime listings & offers
* ⚡ **[Open in Flipkart]({flipkart_url})** — Check live seller discounts & stock
* 🔍 **[Compare on Google Shopping]({google_shop_url})** — Compare prices across all Indian retailers
* 👗 **[Find on Myntra]({myntra_url})** — Browse apparel, footwear & lifestyle catalog
"""


# ---------------------------------------------------------
# 4. Core Multimodal Inference Pipeline
# ---------------------------------------------------------
def run_pipeline(image_input, text_input, audio_input):
    transcribed_text = ""
    audio_output_path = None

    try:
        # A. Voice Transcription (Whisper)
        if audio_input is not None:
            transcription_result = whisper_model.transcribe(audio_input)
            transcribed_text = transcription_result.get("text", "").strip()

        # B. Aggregate Query
        user_query = ""
        if transcribed_text:
            user_query += f"Spoken Query: {transcribed_text}. "
        if text_input and text_input.strip():
            user_query += f"Text Instructions: {text_input.strip()}"

        if not user_query.strip():
            user_query = "Identify the product in this image and provide styling or complementary product recommendations."

        # C. Prepare Multimodal Payload
        contents = []
        if image_input is not None:
            contents.append(image_input)
        contents.append(
            f"You are an expert AI e-commerce personal shopper. Analyze this product photo or receipt and answer the customer query.\n"
            f"Query: {user_query}\n"
            f"Identify the exact item name, category, realistic price in INR (₹), specifications, and helpful styling advice."
        )

        # D. LLM Inference via Google GenAI SDK
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProductMetadata,
                temperature=0.3
            )
        )

        # E. Parse Output Data
        data = json.loads(response.text)
        product_name = data.get("product_name", "Identified Product")
        category = data.get("category", "General")
        price = data.get("estimated_price", "N/A")
        features = data.get("key_features", [])
        summary = data.get("recommendation_summary", "")

        features_bullets = "\n".join([f"- **{f}**" for f in features])

        # F. Format Visual Markdown Cards
        markdown_result = f"""
## 📦 {product_name}
**Category:** `{category}` | **Estimated Market Price:** `{price}`

### 💡 Assistant Recommendations & Analysis
{summary}

### ✨ Key Detected Specifications
{features_bullets}
"""
        # Append dynamic marketplace buttons
        markdown_result += generate_store_cards(product_name)

        # G. Text-to-Speech (gTTS)
        speech_text = f"We identified the {product_name}. {summary}"
        tts = gTTS(text=speech_text[:280], lang="en", slow=False)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        audio_output_path = temp_audio.name

        return (
            markdown_result,
            data,
            audio_output_path,
            transcribed_text if transcribed_text else "(No voice input recorded)"
        )

    except Exception as err:
        error_card = f"""
### ❌ Processing Error
```text
{str(err)}
