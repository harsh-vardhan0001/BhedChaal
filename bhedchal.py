import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
from itertools import count
import os
from datetime import datetime

# Load YOLOv8 model (person detection)
model = YOLO("yolov8n.pt")  # small & fast

st.title("🐑 BhedChaal Crowd Monitor (Web App)")

st.sidebar.title("Settings")
source_option = st.sidebar.radio("Select video source", ("Webcam", "Upload Video"))

if source_option == "Upload Video":
    video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
else:
    video_file = None

# Function to draw heatmap
def draw_heatmap(frame, points): 
    heatmap = np.zeros((frame.s0. hape[0], frame.shape[1]), dtype=np.float32)
    for (x, y) in points:
        cv2.circle(heatmap, (int(x), int(y)), 25, 1, -1)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# Detect abnormal movement
def detect_abnormal_behavior(prev_centers, curr_centers, threshold=50):
    if not prev_centers:
        return False
    movements = [np.linalg.norm(np.array(c) - np.array(p)) for p, c in zip(prev_centers, curr_centers)]
    return any(m > threshold for m in movements)        
                                        
def main():
    stframe = st.empty()
    prev_centers = []
    
    if source_option == "Upload Video":
        if not video_file:
            st.warning("Please upload a video file.")
            return
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video ended or camera not accessible.")
            break

        results = model(frame)[0]
        people_boxes = [box for box in results.boxes.data if int(box[5]) == 0]

        centers = []
        for box in people_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            centers.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
        
        frame = draw_heatmap(frame, centers)

        # Display info
        cv2.putText(frame, f"People Count: {len(centers)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        if detect_abnormal_behavior(prev_centers, centers):
            cv2.putText(frame, "Abnormal Movement Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if len(centers) > 15 and detect_abnormal_behavior(prev_centers, centers):
            cv2.putText(frame, "STAMPED RISK!!!", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        prev_centers = centers

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # Small delay
        if source_option == "Upload Video":
            time.sleep(1/30)  # assuming 30 FPS
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

if __name__ == "__main__":
    main()
    
