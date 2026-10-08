import streamlit as st
import cv2
from xlsxwriter import *
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

# 2. 画面の左側（サイドバー）に信頼度の調整スライダーを設置
conf_score = st.sidebar.slider(
    "AIの判定基準（厳しさ）を調整してください", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)
st.sidebar.write("※数値を大きくするほど、あやしい枠が消えて厳しく判定します。")

st.write("「Start」ボタンを押すとカメラが起動します。左側のスライダーでリアルタイムに枠の調整が可能です！")

# 3. サーバー上で映像を1フレームずつ処理するクラス
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.conf = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=self.conf)
        for r in results:
            img = r.plot()
        return frame.from_ndarray(img, format="bgr24")

# 4. ブラウザ上にリアルタイムカメラを設置（エラーの原因だった通信設定を丸ごと削除し、最もシンプルにしました）
ctx = webrtc_streamer(
    key="cap-detection", 
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

# 5. スライダーの数値をリアルタイムにカメラ処理クラスへ届ける設定
if ctx.video_processor:
    ctx.video_processor.conf = conf_score
