import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

st.set_page_config(page_title="リアルタイムAIカメラ", layout="centered")
st.title("🥤 ペットボトルキャップ認識AI（リアルタイム世界公開版）")

# 1. AIモデルの読み込み
@st.cache_resource
def load_model():
    if os.path.exists("./best.pt"):
        return YOLO("./best.pt")
    else:
        return YOLO("./runs/detect/train/weights/best.pt")

model = load_model()

st.write("「Start」ボタンを押すとカメラが起動し、リアルタイムでボックスが表示されます！")

# 2. サーバー上で映像を1フレームずつ処理するクラス
class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        # 映像をOpenCV形式（NumPy配列）に変換
        img = frame.to_ndarray(format="bgr24")
        
        # AIでキャップを検出（信頼度は0.5に設定）
        results = model(img, conf=0.5)
        
        # 検出結果のボックスを映像に描き込む
        for r in results:
            img = r.plot()
            
        return frame.from_ndarray(img, format="bgr24")

# 3. ブラウザ上にリアルタイムカメラを設置（スマホ等でも動くようにメディア設定を有効化）
webrtc_streamer(
    key="cap-detection", 
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
