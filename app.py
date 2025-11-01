import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import os
import io
import google.generativeai as genai

# ----------------------------
# CONFIGURE GEMINI AI
# ----------------------------
# ⚠️ Replace this with your own key (keep it private)
GOOGLE_API_KEY = "AIzaSyA2ug2CebXrODMdhgym00I4X1EYBqAxlcM"
genai.configure(api_key=GOOGLE_API_KEY)

# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(page_title="🌿 Smart Weed Detection & Advisory", layout="wide")
st.title("🌾 AI-Powered Weed Detection & Farming Assistant")

st.markdown("""
Upload a farm image to detect weeds using a **YOLO-based model**, and get AI-generated
recommendations on **weed management and prevention** using **AI**.
""")

# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

MODEL_PATH = "best.pt"  # local model file
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at `{MODEL_PATH}`. Please place it in your project folder.")
    st.stop()

model = load_model(MODEL_PATH)

# ----------------------------
# FILE UPLOAD SECTION
# ----------------------------
uploaded_file = st.file_uploader("📸 Upload an image of your field", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Detecting weeds..."):
        results = model.predict(image, conf=0.25)
    
    res_img = results[0].plot()
    st.image(res_img, caption="Detection Results", use_container_width=True)

    boxes = results[0].boxes

    # ----------------------------
    # DISPLAY DETECTION RESULTS
    # ----------------------------
    detected_weeds = []
    if boxes:
        st.subheader("🌱 Detected Weeds")
        for i, box in enumerate(boxes):
            cls = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls]
            detected_weeds.append(label)
            st.write(f"• {label} (Confidence: {conf:.2f})")
    else:
        st.warning("No weeds detected. Try another image.")

    # ----------------------------
    # AGENTIC AI ASSISTANT
    # ----------------------------
    if detected_weeds:
        st.subheader("🤖 AI Farming Assistant")

        weed_summary = ", ".join(set(detected_weeds))
        st.write(f"AI is analyzing weed types: **{weed_summary}** ...")

        # Compose AI prompt
        prompt = f"""
        You are an agricultural expert AI assistant helping farmers manage weeds.
        The following weeds were detected in the uploaded field image: {weed_summary}.

        Provide a detailed but practical explanation including:
        1. The typical causes of these weeds.
        2. Organic control methods.
        3. Safe chemical control options (if applicable).
        4. Preventive agricultural practices for future weed reduction.
        5. Additional tips for sustainable weed management.

        Respond in a clear, conversational, and helpful tone for small-scale farmers.
        """

        try:
            model_ai = genai.GenerativeModel("gemini-2.0-flash")
            response = model_ai.generate_content(prompt)
            st.success("✅ AI Response:")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"⚠️ AI assistant failed to generate advice: {e}")

st.markdown("---")
st.caption("Developed by Team OOPS - Weed Detection & Advisory System")
