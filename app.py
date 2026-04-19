import streamlit as st
import cv2
import numpy as np
import pytesseract

st.title("License Plate Recognition (Video URL) 🚗")

video_url = st.text_input("Enter Video URL (mp4 or stream link)")

if video_url:

    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        st.error("Cannot open video stream. Check the URL.")
    else:
        st.success("Processing video...")

        frame_count = 0
        results = []

        frame_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # process every 10th frame (performance control)
            if frame_count % 10 == 0:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.threshold(
                    blur, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]

                text = pytesseract.image_to_string(thresh)

                if text.strip():
                    results.append(text.strip())

            # show live frame in UI
            frame_placeholder.image(frame, channels="BGR")

        cap.release()

        st.subheader("Detected Text")
        st.write(results)