import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import hf_hub_download   # ✅ 허깅페이스 허브에서 다운로드

st.set_page_config(page_title="OCT AI Demo", layout="wide")

# =======================
# Load Trained Model
# =======================
@st.cache_resource
def load_trained_model(model_name):
    repo_id = "Daehwan-shin/oct-ai-models"  # ✅ Hugging Face repo ID

    if model_name == "DenseNet201":
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="densenet201_finetune_best.h5"  # 허깅페이스에 업로드한 파일명
        )
        model = load_model(model_path, compile=False)
        return model, (224, 224), ["CNV / Wet AMD", "DME", "DRUSEN", "NORMAL"]

    else:  # EfficientNet-B4
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="efficientnetb4_finetune_best.h5"
        )
        model = load_model(model_path, compile=False)
        return model, (380, 380), ["CNV / Wet AMD", "DME", "DRUSEN", "NORMAL"]

# =======================
# Grad-CAM
# =======================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def compute_heatmap(self, image_array, class_idx=None):
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
        heatmap = heatmap.numpy()
        heatmap = np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return superimposed

# =======================
# Streamlit UI
# =======================
st.title("🖥️ OCT Image AI Demo (4-Class)")
st.write("DenseNet201 vs EfficientNet-B4 기반 OCT 분류 (CNV / DRUSEN / NORMAL / DME) + Grad-CAM")

model_choice = st.selectbox("모델 선택", ["DenseNet201", "EfficientNet-B4"])
model, img_size, class_labels = load_trained_model(model_choice)

uploaded_file = st.file_uploader("OCT 이미지 업로드", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # OpenCV로 이미지 읽기
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="📷 Uploaded OCT", use_container_width=True)

    # 전처리
    image_resized = cv2.resize(image, img_size)
    image_arr = img_to_array(image_resized)
    image_arr = np.expand_dims(image_arr, axis=0)

    # 모델별 전처리 적용
    if model_choice == "DenseNet201":
        from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre
        image_arr = densenet_pre(image_arr)
        last_conv_layer = "conv5_block32_concat"
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_pre
        image_arr = effnet_pre(image_arr)
        last_conv_layer = "top_conv"

    # 예측
    pred = model.predict(image_arr)
    class_idx = int(np.argmax(pred[0]))
    prob = float(np.max(pred[0]))
    label = class_labels[class_idx]

    st.metric("Prediction", f"{label}", f"{prob:.2f}")

    # 확률 분포 시각화
    import pandas as pd
    probs = pred[0]
    df = pd.DataFrame({
        "class": class_labels,
        "probability": probs
    })
    st.subheader("Class Probabilities")
    st.dataframe(df)
    st.bar_chart(df.set_index("class"))
    
    # Grad-CAM
    st.subheader("Grad-CAM Visualization")
    gradcam = GradCAM(model, last_conv_layer)
    heatmap = gradcam.compute_heatmap(image_arr, class_idx=class_idx)
    overlay_img = gradcam.overlay_heatmap(heatmap, image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original OCT", use_container_width=True)
    with col2:
        st.image(overlay_img, caption=f"Grad-CAM ({label})", use_container_width=True)
