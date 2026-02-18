import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO

st.set_page_config(page_title="Webcam Hand Object Detection", layout="wide")
st.title("🎥 Webcam Hand Object Detection (YOLOv8) — Streamlit")

MODEL_PATH = st.sidebar.text_input("Model path (.pt)", "yolov8n.pt")
CONF = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
IMGSZ = st.sidebar.selectbox("imgsz", [320, 480, 640, 800], index=2)

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

model = load_model(MODEL_PATH)

class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        res = self.model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        annotated = res.plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="yolo-webcam",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=YOLOVideoProcessor,
)
