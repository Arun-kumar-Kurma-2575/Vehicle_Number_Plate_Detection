################################################ Refined Final Image code ###############################################################

from PIL import Image
import cv2
import easyocr
import streamlit as st
import numpy as np
import os
from datetime import datetime


# Create folder for saving plates
output_folder = "plates"
os.makedirs(output_folder, exist_ok=True)
# Generate a timestamped filename

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Streamlit App Interface
st.title("Vehicle Number Plate Detection")

# Upload image for plate detection
uploaded_image = st.file_uploader("Upload an Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Load image using PIL and convert to OpenCV format
    image = Image.open(uploaded_image)
    image = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # Convert to BGR (for OpenCV) if needed
    # if len(image.shape) == 3 and image.shape[2] == 4:  # If image has alpha channel (RGBA)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Load Haar cascade
    vehicle_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    # Detect plates
    plates = vehicle_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=6)


    st.write(f"### Number of Plates Detected: **{len(plates)}**")

    # If no plates detected, notify the user
    if len(plates) == 0:
        st.warning("No number plates were detected in the image.")

    # Process detected plates
    detected_plate_texts = []
    c=1
    for (x,y,w,h) in plates:
        # Draw rectangle around the detected plate
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Ensure valid cropping
        # y1, y2 = max(0, y+7), min(y+h-7, image.shape[0])
        # x1, x2 = max(0, x+7), min(x+w-7, image.shape[1])
        # plate_area = image[y1:y2, x1:x2]
        plate_area=image[y+7:y+h-7,x+7:x+w-7]

        # Perform OCR
        results = reader.readtext(plate_area)

        # Extract text (handling case where OCR detects nothing)
        plate_text = "Unknown"
        if results:
            plate_text = results[0][1]  # Extract text from first result
        
        detected_plate_texts.append(plate_text)

        # Display detected plate
        st.image(plate_area, caption=f"Detected Plate {c}", use_column_width=True)
        # plate_filename = os.path.join(output_folder, f"plate_{c}.jpg")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        plate_filename = os.path.join(output_folder, f"plate_{c}_{timestamp}.jpg")
        cv2.imwrite(plate_filename, plate_area)
        cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        st.write(f"**Detected Plate {c}:** {plate_text}")
        c+=1

    # Display final image with detected plates
    st.image(image, caption="Image with Detected Number Plates", use_column_width=True)


