################################################ Refined Final Video code ###############################################################
import streamlit as st
import cv2
import easyocr
import numpy as np
import os
import hashlib
from datetime import datetime
import tempfile

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Load Haar cascade for number plate detection
vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Create folder for saving plates
output_folder = "plates"
os.makedirs(output_folder, exist_ok=True)

# Initialize variables
frame_count = 0
plate_count = 0
plate_hashes = set()
last_detected_text = ""

# Streamlit UI
st.title("License Plate Detection")
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

# Function to compute hash of an image
def compute_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

# If video is uploaded
if video_file is not None:
    st.video(video_file)

    # Create a temporary file to store video data
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_file.read())
        video_path = temp_file.name

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Couldn't open video file.")
        st.stop()

    # Streamlit placeholder for video
    video_placeholder = st.empty()

    # Stop button
    stop_button = st.button("Stop Video Processing")

    while cap.isOpened():
        has_frame, frame = cap.read()
        frame_count += 1
        if not has_frame:
            break

        # Convert frame to grayscale for better plate detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect plates using Haar cascade
        plates = vehicle_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in plates:
            #frame_count += 1
            if frame_count % 30 != 0:  # Process every 5th frame
                continue
            if (w*h>=8000 or w*h<=2000):
                continue
            st.write(w*h)
            st.write(frame_count)


            # Draw rectangle around detected plate
            cv2.rectangle(frame,(x, y), (x + w, y + h), (255, 0, 0), 3)

            # Extract number plate area
            plate_area = frame[max(0, y + 7): min(y + h - 7, frame.shape[0]),
                               max(0, x + 7): min(x + w - 7, frame.shape[1])]

            if plate_area.size == 0:
                continue
            
            # Convert plate to grayscale
            plate_gray = cv2.cvtColor(plate_area, cv2.COLOR_BGR2GRAY)
            plate_hash = compute_image_hash(plate_gray)

            # Avoid duplicate plate detection using hash
            if plate_hash in plate_hashes:
                continue

            # OCR Processing
            results = reader.readtext(plate_area)
            plate_text = " "
            for (_, text, _) in results:
                plate_text = text.strip()

            # Skip if the same text was detected previously
            if plate_text == last_detected_text:
                continue

            last_detected_text = plate_text

            # Save unique plate image every 10 frames or if a new plate is detected
            if frame_count % 30 == 0 or plate_text != last_detected_text:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plate_filename = os.path.join(output_folder, f"plate_{plate_count}_{timestamp}.jpg")
                cv2.imwrite(plate_filename, plate_area)
                plate_count += 1
                plate_hashes.add(plate_hash)  # Store hash to avoid duplicates

            # Display detected text on frame
            cv2.putText(frame, plate_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame in Streamlit
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Stop processing if the stop button is clicked
        if stop_button:
            #st.write(frame_count)
            st.warning("Video processing stopped.")
            break

    cap.release()

else:
    st.info("Please upload a video file to start processing.")


