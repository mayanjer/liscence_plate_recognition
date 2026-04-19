import streamlit as st
import cv2
import numpy as np
import tempfile
import pytesseract

st.title("Video License Plate Recognition 🚗")

uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_count = 0
    results = []

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every nth frame (performance optimization)
        if frame_count % 10 == 0:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            text = pytesseract.image_to_string(thresh)

            if text.strip():
                results.append(text.strip())

        # Show video frame in Streamlit (optional)
        stframe.image(frame, channels="BGR")

    cap.release()

    st.subheader("Detected Plates / Text:")
    st.write(results)