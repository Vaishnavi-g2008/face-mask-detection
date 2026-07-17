import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------ Page Title ------------------
st.title("😷 Face Mask Detection")

# ------------------ Session State ------------------
if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

# ------------------ Load TFLite Model ------------------
interpreter = tf.lite.Interpreter(model_path="mask_final_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------ Prediction Function ------------------
def predict_mask(img):

    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])
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
    img = Image.open(uploaded_file).convert("RGB")

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
        img = Image.open(camera_image).convert("RGB")

        st.image(img, caption="Captured Image", use_container_width=True)

        predict_mask(img)

        st.session_state.open_camera = False
