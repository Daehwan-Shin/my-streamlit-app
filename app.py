import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import hf_hub_download
import pandas as pd

st.set_page_config(page_title="OCT AI Demo", layout="wide")

# =======================
# Load Trained Model
# =======================
@st.cache_resource
def load_trained_model(model_name):
    repo_id = "Daehwan-shin/oct-ai-models"

    if model_name == "DenseNet201":
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="densenet201_4class_finetune_opt_best.h5"
        )
        model = load_model(model_path, compile=False)
        return model, (224, 224), ["CNV / Wet AMD", "DME", "DRUSEN", "NORMAL"], "conv5_block32_concat"

    else:  # EfficientNet-B4
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="efficientnetb4_4class_finetune_opt_best.h5"
        )
        model = load_model(model_path, compile=False)
        return model, (380, 380), ["CNV / Wet AMD", "DME", "DRUSEN", "NORMAL"], "top_conv"

# =======================
# Grad-CAM & Grad-CAM++
# =======================
class XAIVisualizer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def gradcam(self, image_array, class_idx=None):
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(self.target_layer).output, self.model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap.numpy(), 0)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap

    def gradcam_plus_plus(self, image_array, class_idx=None):
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(self.target_layer).output, self.model.output]
        )
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    conv_outputs, predictions = grad_model(image_array)
                    if class_idx is None:
                        class_idx = tf.argmax(predictions[0])
                    loss = predictions[:, class_idx]
                grads = tape3.gradient(loss, conv_outputs)
            grads2 = tape2.gradient(grads, conv_outputs)
        grads3 = tape1.gradient(grads2, conv_outputs)

        conv_outputs = conv_outputs[0].numpy()
        grads = grads[0].numpy()
        grads2 = grads2[0].numpy()
        grads3 = grads3[0].numpy()

        numerator = grads2
        denominator = 2.0 * grads2 + grads3 * conv_outputs
        denominator = np.where(denominator != 0.0, denominator, np.ones_like(denominator))

        alphas = numerator / denominator
        alphas = np.maximum(alphas, 0)
        weights = np.sum(alphas * np.maximum(grads, 0), axis=(0, 1))

        heatmap = np.sum(weights * conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

# =======================
# Streamlit UI
# =======================
st.title("🖥️ OCT Image AI Demo (4-Class)")
st.write("DenseNet201 vs EfficientNet-B4 기반 OCT 분류 (CNV / DME / DRUSEN / NORMAL) + Grad-CAM/Grad-CAM++")

model_choice = st.selectbox("모델 선택", ["DenseNet201", "EfficientNet-B4"])
model, img_size, class_labels, last_conv_layer = load_trained_model(model_choice)

uploaded_file = st.file_uploader("OCT 이미지 업로드", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="📷 Uploaded OCT", use_container_width=True)

    # Preprocess
    image_resized = cv2.resize(image, img_size)
    image_arr = img_to_array(image_resized)
    image_arr = np.expand_dims(image_arr, axis=0)

    if model_choice == "DenseNet201":
        from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre
        image_arr = densenet_pre(image_arr)
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_pre
        image_arr = effnet_pre(image_arr)

    # Prediction
    pred = model.predict(image_arr)
    class_idx = int(np.argmax(pred[0]))
    prob = float(np.max(pred[0]))
    label = class_labels[class_idx]

    st.metric("Prediction", f"{label}", f"{prob:.2f}")

    # Probability distribution
    probs = pred[0]
    df = pd.DataFrame({
        "class": class_labels,
        "probability": probs
    })
    st.subheader("Class Probabilities")
    st.dataframe(df.style.format({"probability": "{:.2f}"}))
    st.bar_chart(df.set_index("class"))

    # Grad-CAM & Grad-CAM++
    st.subheader("Explainability Visualization")
    xai = XAIVisualizer(model, last_conv_layer)
    heatmap_cam = xai.gradcam(image_arr, class_idx=class_idx)
    heatmap_campp = xai.gradcam_plus_plus(image_arr, class_idx=class_idx)

    overlay_cam = xai.overlay_heatmap(heatmap_cam, image)
    overlay_campp = xai.overlay_heatmap(heatmap_campp, image)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original OCT", use_container_width=True)
    with col2:
        st.image(overlay_cam, caption=f"Grad-CAM ({label})", use_container_width=True)
    with col3:
        st.image(overlay_campp, caption=f"Grad-CAM++ ({label})", use_container_width=True)
