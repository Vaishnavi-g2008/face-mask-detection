import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("😷 Face Mask Detection")

# Session State
if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="mask_final_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Upload Image
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])
    prob = prediction[0][0]

    if prob > 0.5:
        st.success(f"Prediction: WITHOUT Mask ❌😷 ({prob:.2%})")
    else:
        st.success(f"Prediction: WITH Mask ✅😷 ({(1-prob):.2%})")

# Camera Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("📸 Open Camera"):
        st.session_state.open_camera = True

with col2:
    if st.button("❌ Close Camera"):
        st.session_state.open_camera = False

# Camera Input
if st.session_state.open_camera:

    camera_image = st.camera_input("Click Photo")

    if camera_image is not None:

        img = Image.open(camera_image).convert("RGB")
        img = img.resize((128, 128))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]["index"])
        confidence = prediction[0][0]

        if confidence > 0.5:
            st.success(f"Without Mask ❌😷 ({confidence:.2%})")
        else:
            st.success(f"With Mask ✅😷 ({(1-confidence):.2%})")

        st.session_state.open_camera = False
