import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ------------------ Page Title ------------------
st.title("😷 Face Mask Detection")

# ------------------ Session State ------------------
if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

# ------------------ Load Model ------------------
model = load_model("mask_final_quant.tflite")

# ------------------ Prediction Function ------------------
def predict_mask(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    confidence = prediction[0][0]

    if confidence > 0.5:
        st.error(f"❌ WITHOUT Mask ({confidence:.2%})")
    else:
        st.success(f"✅ WITH Mask ({(1-confidence):.2%})")

# ------------------ Upload Image ------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    predict_mask(img)

st.markdown("---")

# ------------------ Camera Buttons ------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("📸 Open Camera"):
        st.session_state.open_camera = True

with col2:
    if st.button("❌ Close Camera"):
        st.session_state.open_camera = False

# ------------------ Camera ------------------
if st.session_state.get("open_camera", False):

    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        img = Image.open(camera_image)

        st.image(img, caption="Captured Image", use_container_width=True)

        predict_mask(img)

        # Optional: Close camera automatically after prediction
        st.session_state.open_camera = False
