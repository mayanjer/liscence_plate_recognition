import streamlit as st
import cv2
import numpy as np
import pytesseract
import imutils
import tempfile
import os
import re
from skimage import measure

# --- CORE LOGIC (PORTED FROM YOUR NOTEBOOK) ---

def sort_cont(character_contours):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours,
                                                          boundingBoxes),
                                                      key=lambda b: b[1][i],
                                                      reverse=False))
    return character_contours

def segment_chars(plate_img, fixed_width):
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]
    thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    labels = measure.label(thresh, background=0)
    charCandidates = np.zeros(thresh.shape, dtype='uint8')
    characters = []

    for label in np.unique(labels):
        if label == 0: continue
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            aspectRatio = boxW / float(boxH) if boxH != 0 else 0
            solidity = cv2.contourArea(c) / float(boxW * boxH) if (boxW * boxH) != 0 else 0
            heightRatio = boxH / float(plate_img.shape[0]) if plate_img.shape[0] != 0 else 0
            if aspectRatio < 1.0 and solidity > 0.15 and 0.5 < heightRatio < 0.95 and boxW > 14:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    contours, _ = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sort_cont(contours)
        addPixel = 4
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            y_start = max(0, y - addPixel)
            x_start = max(0, x - addPixel)
            temp = bgr_thresh[y_start:y + h + (addPixel * 2), x_start:x + w + (addPixel * 2)]
            characters.append(temp)
        return characters
    return None

class PlateFinder:
    def __init__(self, minPlateArea, maxPlateArea):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea

    def preprocess(self, input_img):
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) if len(input_img.shape) == 3 else input_img
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, 50, 150)
        _, thresh = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(edges, thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    def extract_contours(self, after_preprocess):
        contours, _ = cv2.findContours(after_preprocess, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def check_plate(self, input_img, contour):
        x, y, w, h = cv2.boundingRect(contour)
        aspect = float(w) / h if h != 0 else 0
        if 2.0 < aspect < 8.0 and w > 20 and h > 10:
            plate_img = input_img[y:y + h, x:x + w]
            chars = segment_chars(plate_img, 400)
            return plate_img, chars, (x, y, w, h)
        return None, None, None

    def find_possible_plates(self, input_img):
        self.after_preprocess = self.preprocess(input_img)
        contours = self.extract_contours(self.after_preprocess)
        plates_found = []
        self.char_on_plate = []
        self.plate_coords = []
        
        for cnt in contours:
            plate, chars, coords = self.check_plate(input_img, cnt)
            if plate is not None:
                plates_found.append(plate)
                self.char_on_plate.append(chars)
                self.plate_coords.append(coords)
        return plates_found if plates_found else None

class OCR:
    def __init__(self):
        self.char_config = r"--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.plate_config = r"--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.whitelist_pattern = re.compile(r"[^A-Z0-9]")

    def _clean_text(self, text):
        return self.whitelist_pattern.sub("", text.upper().strip())

    def label_plate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (400, 100), interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config=self.plate_config)
        return self._clean_text(text)

    def label_image_list(self, listImages):
        plate = ""
        for img in listImages:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            char = pytesseract.image_to_string(thresh, config=self.char_config)
            clean_char = self._clean_text(char)
            if clean_char: plate += clean_char
        return plate, len(plate)

# --- STREAMLIT UI ---

st.set_page_config(page_title="ALPR Uganda", layout="wide")

st.title("🚗 Automated License Plate Recognition")
st.markdown("Upload a vehicle video to scan for license plates in real-time.")

# Sidebar Settings
st.sidebar.header("Processing Controls")
frame_skip = st.sidebar.slider("Frame Skip", 1, 15, 5)
min_area = st.sidebar.number_input("Min Plate Area", value=100)
max_area = st.sidebar.number_input("Max Plate Area", value=50000)

# Initialize Engines
plate_finder = PlateFinder(min_area, max_area)
ocr = OCR()

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st.subheader("Detected Plates")
        st_list = st.empty()

    detected_plates = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % frame_skip != 0: continue

        frame_resized = cv2.resize(frame, (640, 480))
        plates = plate_finder.find_possible_plates(frame_resized)

        if plates:
            for idx, plate in enumerate(plates):
                chars = plate_finder.char_on_plate[idx]
                # Draw bounding box for visual feedback
                x, y, w, h = plate_finder.plate_coords[idx]
                cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Recognition logic
                if chars:
                    p_text, length = ocr.label_image_list(chars)
                else:
                    p_text = ocr.label_plate(plate)
                    length = len(p_text)

                if length >= 3:
                    detected_plates.add(p_text)

        # Update Streamlit UI
        st_frame.image(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), channels="RGB")
        st_list.write(", ".join(list(detected_plates)))

    cap.release()
    os.unlink(tfile.name)
    st.success("Processing Complete!")