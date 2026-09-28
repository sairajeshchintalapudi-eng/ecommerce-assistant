import os
import json
import tempfile
import urllib.parse
from typing import List
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

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Warning: Client initialization failed: {e}")

print("Loading Whisper ASR model ('tiny')...")
whisper_model = whisper.load_model("tiny", device="cpu")


# ---------------------------------------------------------
# 2. Pydantic Structured Output Schema
# ---------------------------------------------------------
class ProductMetadata(BaseModel):
    product_name: str = Field(description="Specific brand and product title identified")
    category: str = Field(description="Category, e.g., Footwear, Electronics, Apparel")
    estimated_price: str = Field(description="Estimated price or price range in INR (₹)")
    key_features: List[str] = Field(description="Key attributes or specifications")
    recommendation_summary: str = Field(description="Styling advice or product breakdown")


# ---------------------------------------------------------
# 3. Dynamic Marketplace Link Generator
# ---------------------------------------------------------
def generate_store_cards(item_name: str) -> str:
    safe_query = urllib.parse.quote_plus(item_name if item_name else "product")
    amazon_url = f"https://www.amazon.in/s?k={safe_query}"
    flipkart_url = f"https://www.flipkart.com/search?q={safe_query}"
    google_shop_url = f"https://www.google.com/search?tbm=shop&q={safe_query}"
    myntra_url = f"https://www.myntra.com/{safe_query}"

    return (
        "\n\n---\n"
        "### 🛍️ Direct Store & Purchase Links\n"
        f"* 🛒 **[Open in Amazon India]({amazon_url})** — Search Prime listings & offers\n"
        f"* ⚡ **[Open in Flipkart]({flipkart_url})** — Check live seller discounts & stock\n"
        f"* 🔍 **[Compare on Google Shopping]({google_shop_url})** — Compare prices across Indian retailers\n"
        f"* 👗 **[Find on Myntra]({myntra_url})** — Browse apparel & lifestyle catalog\n"
    )


# ---------------------------------------------------------
# 4. Core Multimodal Inference Pipeline
# ---------------------------------------------------------
def run_pipeline(image_input, text_input, audio_input):
    transcribed_text = ""
    audio_output_path = None

    try:
        if not client:
            raise ValueError("GEMINI_API_KEY is missing or invalid in Render Environment settings.")

        # Voice Transcription (Whisper)
        if audio_input is not None:
            transcription_result = whisper_model.transcribe(audio_input)
            transcribed_text = transcription_result.get("text", "").strip()

        # Build Query
        user_query = ""
        if transcribed_text:
            user_query += f"Spoken Query: {transcribed_text}. "
        if text_input and text_input.strip():
            user_query += f"Text Instructions: {text_input.strip()}"

        if not user_query.strip():
            user_query = "Identify the product in this image and provide styling or complementary product recommendations."

        # Prepare Payload
        contents = []
        if image_input is not None:
            contents.append(image_input)
        contents.append(
            f"You are an expert AI e-commerce personal shopper. Analyze this product photo or receipt and answer the customer query.\n"
            f"Query: {user_query}\n"
            f"Identify the exact item name, category, realistic price in INR (₹), specifications, and helpful styling advice."
        )

        # Gemini Inference
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProductMetadata,
                temperature=0.3
            )
        )

        data = json.loads(response.text)
        product_name = data.get("product_name", "Identified Product")
        category = data.get("category", "General")
        price = data.get("estimated_price", "N/A")
        features = data.get("key_features", [])
        summary = data.get("recommendation_summary", "")

        features_bullets = "\n".join([f"- **{f}**" for f in features])

        markdown_result = (
            f"## 📦 {product_name}\n"
            f"**Category:** `{category}` | **Estimated Market Price:** `{price}`\n\n"
            f"### 💡 Assistant Recommendations & Analysis\n{summary}\n\n"
            f"### ✨ Key Specifications\n{features_bullets}"
        )
        markdown_result += generate_store_cards(product_name)

        # Text-to-Speech (gTTS)
        speech_text = f"We identified the {product_name}. {summary}"
        tts = gTTS(text=speech_text[:250], lang="en", slow=False)
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
        error_card = f"### ❌ Processing Error\n\n**Details:** {str(err)}\n\n*Please verify your Gemini API key and try again.*"
        return error_card, {}, None, transcribed_text


# ---------------------------------------------------------
# 5. Interactive Gradio User Interface
# ---------------------------------------------------------
with gr.Blocks(title="AI Multimodal E-Commerce Assistant", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")) as demo:
    gr.Markdown(
        "# 🛒 Multimodal Smart E-Commerce Assistant\n"
        "*Upload any product photo or receipt, or speak your query to receive product discovery and store links.*"
    )

    with gr.Row():
        with gr.Column(scale=5):
            image_input = gr.Image(
                label="1. Upload Product Photo or Receipt",
                type="pil",
                sources=["upload", "clipboard", "webcam"]
            )
            text_input = gr.Textbox(
                label="2. Customer Text Query",
                placeholder="e.g., 'Find matching shoes for this outfit' or 'Summarize this receipt'",
                lines=2
            )
            audio_input = gr.Audio(
                label="3. Spoken Voice Request (Microphone)",
                sources=["microphone", "upload"],
                type="filepath"
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 Process & Recommend", variant="primary", scale=2)
                clear_btn = gr.ClearButton(
                    components=[image_input, text_input, audio_input],
                    value="🔄 Clear",
                    scale=1
                )

        with gr.Column(scale=6):
            voice_transcript_output = gr.Textbox(
                label="Transcribed Voice Query (Whisper)",
                interactive=False,
                placeholder="Your speech transcript will appear here..."
            )
            audio_response_output = gr.Audio(
                label="🔊 Spoken Audio Recommendation (gTTS)",
                autoplay=True
            )
            markdown_output = gr.Markdown(
                label="Product Recommendations & Marketplace Links",
                value="*Recommendations and purchase links will populate after processing.*"
            )
            json_output = gr.JSON(
                label="Structured Metadata Schema (JSON)"
            )

    submit_btn.click(
        fn=run_pipeline,
        inputs=[image_input, text_input, audio_input],
        outputs=[markdown_output, json_output, audio_response_output, voice_transcript_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
