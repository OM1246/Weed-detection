import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import io
from datetime import datetime
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# ----------------------------
# CONFIGURE GEMINI AI
# ----------------------------
GOOGLE_API_KEY = "AIzaSyA2ug2CebXrODMdhgym00I4X1EYBqAxlcM"
genai.configure(api_key=GOOGLE_API_KEY)

# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(page_title="🌿 Smart Weed Detection & Advisory", layout="wide")
st.title("🌾 AI-Powered Weed Detection & Farming Assistant")

st.markdown("""
Upload an image from your field to detect weeds using a **YOLO model**.  
Then get **AI-generated recommendations** on how to control and prevent them.  
You can also **download a detailed report** for farm management.
""")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at `{MODEL_PATH}`. Please add it to your project folder.")
    st.stop()

model = load_model(MODEL_PATH)

# ----------------------------
# FILE UPLOAD SECTION
# ----------------------------
uploaded_file = st.file_uploader("📸 Upload a field image", type=["jpg", "jpeg", "png"])

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
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls]
            detected_weeds.append(label)
            st.write(f"• {label} (Confidence: {conf:.2f})")
    else:
        st.warning("No weeds detected. Try another image.")

    # ----------------------------
    # GEMINI AI ASSISTANT
    # ----------------------------
    ai_response_text = None
    if detected_weeds:
        st.subheader("🤖 AI Farming Assistant")

        weed_summary = ", ".join(set(detected_weeds))
        st.write(f"AI analyzing detected weeds: **{weed_summary}** ...")

        prompt = f"""
        You are an agricultural expert AI assistant for weed management.

        Detected weeds: {weed_summary}

        Please provide:
        1. Common causes and conditions for these weeds.
        2. Organic control methods.
        3. Safe chemical treatment options.
        4. Preventive practices for sustainable weed management.
        5. Any region-specific advice for Indian farmers.

        Make the response farmer-friendly and concise.
        """

        try:
            ai_model = genai.GenerativeModel("gemini-2.0-flash")
            response = ai_model.generate_content(prompt)
            ai_response_text = response.text
            st.success("✅ AI Advisory:")
            st.markdown(ai_response_text)
        except Exception as e:
            st.error(f"⚠️ AI failed to generate advice: {e}")

    # ----------------------------
    # REPORT GENERATION
    # ----------------------------
    if ai_response_text and detected_weeds:
        st.subheader("📄 Generate Detailed Report")

        if st.button("🧾 Download PDF Report"):
            # Create PDF in memory
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("<b>🌿 Smart Weed Detection Report</b>", styles["Title"]))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
            elements.append(Paragraph(f"<b>Detected Weeds:</b> {', '.join(set(detected_weeds))}", styles["Normal"]))
            elements.append(Spacer(1, 12))

            # Save and add the annotated image
            img_temp = "detected_image.jpg"
            Image.fromarray(res_img).save(img_temp)
            elements.append(RLImage(img_temp, width=5.5*inch, height=3.5*inch))
            elements.append(Spacer(1, 18))

            elements.append(Paragraph("<b>AI Advisory:</b>", styles["Heading2"]))
            elements.append(Paragraph(ai_response_text.replace("\n", "<br/>"), styles["Normal"]))

            doc.build(elements)
            buffer.seek(0)

            st.download_button(
                label="📥 Download Report as PDF",
                data=buffer,
                file_name="weed_detection_report.pdf",
                mime="application/pdf"
            )

st.markdown("---")
st.caption("👨‍🌾 Developed by Team OOPS — AI-Driven Weed Detection & Advisory System")
