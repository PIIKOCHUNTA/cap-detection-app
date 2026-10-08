import streamlit as st
import cv2
from ultralytics import YOLO
import os

st.set_page_config(page_title="リアルタイムAIカメラ", layout="centered")
st.title("🥤 ペットボトルキャップ認識AI（リアルタイムブラウザテスト）")

# 1. AIモデルの読み込み
@st.cache_resource
def load_model():
    if os.path.exists("./best.pt"):
        return YOLO("./best.pt")
    else:
        return YOLO("./runs/detect/train/weights/best.pt")

model = load_model()

# 2. 判定基準の調整スライダー
conf_score = st.sidebar.slider("AIの厳しさ調整", 0.0, 1.0, 0.5, 0.05)

# 3. ブラウザ上にカメラ入力機能を設置
# ※起動するとブラウザからカメラの許可を求められます
img_file_buffer = st.camera_input("カメラに向かってキャップを映してください！")

if img_file_buffer is not None:
    from PIL import Image
    import numpy as np

    # 撮影された、またはプレビューの画像を取得
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    # AIによる認識を実行
    with st.spinner("AIが判定中..."):
        results = model(img_array, conf=conf_score)

    # 判定結果を表示
    for r in results:
        res_plotted = r.plot()
        st.image(res_plotted, caption="AIの判定結果", use_column_width=True)
        st.success(f"ペットボトルのキャップが **{len(r.boxes)}個** 検出されました！")
