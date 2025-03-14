################################################ Refined Final Live-Stream code  ###############################################################
import streamlit as st
import cv2
import easyocr
import numpy as np
import os
import hashlib
from datetime import datetime

# Initialize OCR reader
reader = easyocr.Reader(['en'])

#Load Haar cascade for number plate detection
vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Create folder for saving plates
output_folder = "plates"
os.makedirs(output_folder, exist_ok=True)
frame_count=-1
plate_hashes = set()
plate_count=0
last_detected_text = ""
# Streamlit UI
st.title("Real-Time License Plate Detection")

# Start video capture (Webcam)
cap = cv2.VideoCapture(0)

# Streamlit placeholders for video
video_placeholder = st.empty()

# Stop button handling
if "stop" not in st.session_state:
    st.session_state.stop = False

if st.button("Stop Video Processing"):
    st.session_state.stop = True

# Function to compute hash of an image
def compute_image_hash(image):
    """Compute a unique hash for an image to prevent duplicates."""
    return hashlib.md5(image.tobytes()).hexdigest()

while cap.isOpened() and not st.session_state.stop:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    # Detect number plates
    plates = vehicle_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in plates:
        frame_count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Extract number plate area
        plate_area = frame[max(0, y+7): min(y+h-7, frame.shape[0]), 
                 max(0, x+7): min(x+w-7, frame.shape[1])]
        
        if plate_area.size == 0:
            continue

        plate_gray = cv2.cvtColor(plate_area, cv2.COLOR_BGR2GRAY)
        plate_hash = compute_image_hash(plate_gray)

        # Check for duplicate image
        if plate_hash in plate_hashes:
            continue
            
        # OCR Processing
        results = reader.readtext(plate_area)
        plate_text = "Unknown"
        for (_, text, _) in results:
            plate_text = text.strip()

        # Skip redundant detections
        if plate_text == last_detected_text:
            continue  

        # Using set avoids duplicates
        last_detected_text = plate_text  # Update last detected text

            # Save unique plate image every 10 frames or if a new plate is detected
        if frame_count % 5 == 0 or plate_text != last_detected_text:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
            plate_filename = os.path.join(output_folder, f"plate_{plate_count}_{timestamp}.jpg")
            cv2.imwrite(plate_filename, plate_area)
            plate_count += 1
            plate_hashes.add(plate_hash)  # Store hash to avoid duplicates

            # Display detected text on frame
        cv2.putText(frame, plate_text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show processed video in Streamlit
    
    video_placeholder.image(frame, channels="BGR", use_column_width=True)

cap.release()