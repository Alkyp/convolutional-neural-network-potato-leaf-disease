import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from PIL import Image
from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Potato Leaf Disease Detection",
    page_icon="🍃",
    layout="centered"
)

# =============================
# LOAD LOTTIE
# =============================
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_leaf = load_lottie_url(
    "https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"
)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/potato_cnn.h5")

model = load_model()
CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight"]

# =============================
# GRAD-CAM FUNCTION
# =============================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# =============================
# HEADER
# =============================
st_lottie(lottie_leaf, height=220)

st.markdown("""
<h1 style='text-align:center;'>🍃 Potato Leaf Disease Detection</h1>
<p style='text-align:center; color:gray;'>
CNN-based Image Classification with Grad-CAM
</p>
""", unsafe_allow_html=True)

st.divider()

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader(
    "📤 Upload gambar daun kentang",
    type=["jpg", "jpeg", "png"]
)

# =============================
# PREDICTION
# =============================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing image..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            confidence = float(np.max(preds))
            class_idx = int(np.argmax(preds))
            label = CLASS_NAMES[class_idx]

        st.success("Prediction Complete 🎉")

        # =============================
        # RESULT CARD
        # =============================
        st.markdown(f"""
        <div style="background:#0f172a;padding:25px;border-radius:15px;text-align:center;">
            <h2 style="color:#22c55e;">{label}</h2>
            <p style="color:#cbd5f5;">Confidence: <b>{confidence*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        # =============================
        # CONFIDENCE BAR CHART
        # =============================
        st.subheader("📊 Prediction Confidence")
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, preds[0])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # =============================
        # GRAD-CAM
        # =============================
        # st.subheader("🔥 Grad-CAM Visualization")

        # last_conv_layer = model.layers[-3].name
        # heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

        # heatmap = cv2.resize(heatmap, (224, 224))
        # heatmap = np.uint8(255 * heatmap)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # original = np.array(img)
        # superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        # st.image(superimposed, caption="Model Focus Area", use_container_width=True)

        # =============================
        # SAVE HISTORY
        # =============================
        history_file = "history.csv"
        data = {
            "time": datetime.now(),
            "label": label,
            "confidence": confidence
        }

        df_new = pd.DataFrame([data])

        if os.path.exists(history_file):
            df_old = pd.read_csv(history_file)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new

        df.to_csv(history_file, index=False)

        # =============================
        # EXPLANATION
        # =============================
        if label == "Healthy":
            st.info("🌱 Daun sehat, tidak terdeteksi penyakit.")
        elif label == "Early Blight":
            st.warning("⚠ Early Blight terdeteksi. Lakukan penanganan dini.")
        else:
            st.error("🚨 Late Blight terdeteksi! Risiko tinggi gagal panen.")

# =============================
# HISTORY SECTION
# =============================
st.divider()
st.subheader("🕒 Prediction History")

if os.path.exists("history.csv"):
    df_history = pd.read_csv("history.csv")
    st.dataframe(df_history, use_container_width=True)
else:
    st.write("Belum ada data prediksi.")

# =============================
# FOOTER
# =============================
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Built with ❤️ using Streamlit, CNN & Grad-CAM<br>
Animation by LottieFiles
</p>
""", unsafe_allow_html=True)
