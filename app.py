import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import hf_hub_download
import pandas as pd
from PIL import Image

# -----------------------------
# Streamlit & TensorFlow setup
# -----------------------------
st.set_page_config(page_title="OCT AI Demo", layout="wide")

# TF GPU memory growth (OOM 완화)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("⚠️ Could not set memory growth:", e)

# -----------------------------
# Load Trained Model (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_trained_model(model_name: str):
    repo_id = "Daehwan-shin/oct-ai-models"

    if model_name == "DenseNet201":
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="densenet201_4class_finetune_cnvC_best.h5"
        )
        model = load_model(model_path, compile=False)
        input_size = (224, 224)
        class_labels = ["CNV / Wet AMD", "DME", "DRUSEN", "NORMAL"]
        # DenseNet 마지막 대형 conv concat 레이어
        target_layer = "conv5_block32_concat"
        # 존재 확인 & 폴백
        try:
            model.get_layer(target_layer)
        except Exception:
            # 가장 마지막 conv 유사 레이어를 탐색
            cand = [l.name for l in model.layers if "concat" in l.name or "conv" in l.name][-1]
            target_layer = cand
        return model, input_size, class_labels, target_layer

    else:  # EfficientNet-B4
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="efficientnetb4_4class_finetune_cnvC_best.h5"
        )
        model = load_model(model_path, compile=False)
        input_size = (380, 380)
        class_labels = ["CNV / Wet AMD", "DME", "DRUSEN", "NORMAL"]
        # CAM 타깃: 너무 마지막 activation보단 skip-add 지점이 안정적인 경우가 많음
        target_layer = None
        for cand in ["block6e_add", "block7a_project_bn", "top_conv"]:
            try:
                model.get_layer(cand)
                target_layer = cand
                break
            except Exception:
                continue
        if target_layer is None:
            # 마지막 conv 비슷한 레이어 하나 폴백
            conv_like = [l.name for l in model.layers if "conv" in l.name or "project" in l.name]
            target_layer = conv_like[-1]
        return model, input_size, class_labels, target_layer

# -----------------------------
# XAI (Grad-CAM / LayerCAM)
# -----------------------------
class XAIVisualizer:
    def __init__(self, model: tf.keras.Model, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        # features + logits 모델
        self.grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.target_layer).output, self.model.output]
        )

    def _ensure_float(self, x):
        if isinstance(x, np.ndarray):
            return tf.convert_to_tensor(x, dtype=tf.float32)
        return tf.cast(x, tf.float32)

    def gradcam(self, image_array, class_idx=None, use_logits=False):
        """
        Classic Grad-CAM: GAP(weights) * feature, ReLU, min-max norm
        image_array: [1,H,W,3] float
        """
        x = self._ensure_float(image_array)
        with tf.GradientTape() as tape:
            conv_feats, preds = self.grad_model(x, training=False)
            if class_idx is None:
                class_idx = tf.argmax(preds[0])
            if use_logits:
                score = preds[:, class_idx]
            else:
                score = tf.nn.softmax(preds, axis=-1)[:, class_idx]

        grads = tape.gradient(score, conv_feats)  # [1,h,w,c]
        if grads is None:
            return None

        # Global Average Pooling of grads -> weights
        pooled = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # [1,1,1,c]
        # Weighted sum
        cam = tf.reduce_sum(tf.nn.relu(conv_feats) * pooled, axis=-1)[0]  # [h,w]

        # Normalize 0~1
        cmin = tf.reduce_min(cam)
        cmax = tf.reduce_max(cam)
        cam = tf.where(cmax > cmin, (cam - cmin) / (cmax - cmin), tf.zeros_like(cam))
        return cam.numpy()

    def layercam(self, image_array, class_idx=None, use_logits=False):
        """
        LayerCAM: ReLU(grads) * ReLU(features), 채널 합산
        """
        x = self._ensure_float(image_array)
        with tf.GradientTape() as tape:
            conv_feats, preds = self.grad_model(x, training=False)
            if class_idx is None:
                class_idx = tf.argmax(preds[0])
            if use_logits:
                score = preds[:, class_idx]
            else:
                score = tf.nn.softmax(preds, axis=-1)[:, class_idx]

        grads = tape.gradient(score, conv_feats)  # [1,h,w,c]
        if grads is None:
            return None
        cam = tf.reduce_sum(tf.nn.relu(grads) * tf.nn.relu(conv_feats), axis=-1)[0]  # [h,w]

        cmin = tf.reduce_min(cam)
        cmax = tf.reduce_max(cam)
        cam = tf.where(cmax > cmin, (cam - cmin) / (cmax - cmin), tf.zeros_like(cam))
        return cam.numpy()

    @staticmethod
    def overlay_heatmap(heatmap01: np.ndarray, image_bgr: np.ndarray, alpha: float = 0.4):
        """
        heatmap01: [h,w] 0~1
        image_bgr: OpenCV BGR
        """
        if heatmap01 is None:
            return None
        h, w = image_bgr.shape[:2]
        hm = cv2.resize(heatmap01, (w, h), interpolation=cv2.INTER_LINEAR)
        hm = np.uint8(np.clip(hm, 0, 1) * 255)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image_bgr, 1 - alpha, hm, alpha, 0)
        return blended

# -----------------------------
# UI
# -----------------------------
st.title("🖥️ OCT Image AI Demo (4-Class)")
st.write("DenseNet201 vs EfficientNet-B4 기반 OCT 분류 (CNV / DME / DRUSEN / NORMAL) + Grad-CAM / LayerCAM")

model_choice = st.selectbox("모델 선택", ["DenseNet201", "EfficientNet-B4"])
model, img_size, class_labels, target_layer = load_trained_model(model_choice)
st.caption(f"🔎 CAM target layer: `{target_layer}`")

uploaded_file = st.file_uploader("OCT 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1) 이미지 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("이미지 로드에 실패했습니다. 다른 파일로 시도하세요.")
        st.stop()

    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="📷 Uploaded OCT", use_column_width=True)

    # 2) 전처리
    image_resized = cv2.resize(image_bgr, img_size, interpolation=cv2.INTER_LINEAR)
    image_arr = img_to_array(image_resized)
    image_arr = np.expand_dims(image_arr, axis=0)

    if model_choice == "DenseNet201":
        from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre
        image_arr = densenet_pre(image_arr)
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_pre
        image_arr = effnet_pre(image_arr)

    # 3) 예측
    pred = model.predict(image_arr, verbose=0)
    class_idx = int(np.argmax(pred[0]))
    prob = float(np.max(pred[0]))
    label = class_labels[class_idx]

    st.metric("Prediction", f"{label}", f"{prob:.2f}")

    # 4) 확률 분포
    probs = pred[0]
    df = pd.DataFrame({"class": class_labels, "probability": probs})
    st.subheader("Class Probabilities")
    st.dataframe(df.style.format({"probability": "{:.2f}"}))
    st.bar_chart(df.set_index("class"))

    # 5) XAI: Grad-CAM & LayerCAM
    st.subheader("Explainability Visualization")
    xai = XAIVisualizer(model, target_layer)

    heatmap_cam = xai.gradcam(image_arr, class_idx=class_idx)
    heatmap_layercam = xai.layercam(image_arr, class_idx=class_idx)

    overlay_cam = XAIVisualizer.overlay_heatmap(heatmap_cam, image_bgr)
    overlay_layercam = XAIVisualizer.overlay_heatmap(heatmap_layercam, image_bgr)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original OCT", use_column_width=True)
    with col2:
        if overlay_cam is not None:
            st.image(cv2.cvtColor(overlay_cam, cv2.COLOR_BGR2RGB), caption=f"Grad-CAM ({label})", use_column_width=True)
        else:
            st.warning("Grad-CAM 계산 실패(레이어 확인 필요)")
    with col3:
        if overlay_layercam is not None:
            st.image(cv2.cvtColor(overlay_layercam, cv2.COLOR_BGR2RGB), caption=f"LayerCAM ({label})", use_column_width=True)
        else:
            st.warning("LayerCAM 계산 실패(레이어/gradient 확인 필요)")
else:
    st.info("좌측 영역에 OCT 이미지를 업로드하세요.")
