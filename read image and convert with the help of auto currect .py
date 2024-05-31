import streamlit as st
import pytesseract
import cv2
import numpy as np
from spellchecker import SpellChecker

# Specify Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to preprocess the image
def preprocess_image(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    # Further denoising using morphological operations
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Invert the image
    inverted_img = cv2.bitwise_not(opening)
    return inverted_img

# Function to perform OCR and spell-checking
def process_image(img):
    # Preprocess the image
    preprocessed_img = preprocess_image(img)
    # Extract text from preprocessed image
    text = pytesseract.image_to_string(preprocessed_img)
    st.write("**Extracted Text:**", text)

    # Perform spell-checking and auto-correction
    spell = SpellChecker()
    corrected_text = []
    for word in text.split():
        # Check if the word is misspelled
        if spell.unknown([word]):
            # Get the most likely correction
            correction = spell.correction(word)
            corrected_text.append(correction)
        else:
            corrected_text.append(word)
    # Join the corrected words back into a sentence
    corrected_text = ' '.join([word for word in corrected_text if word is not None])

    st.write("**Corrected Text:**", corrected_text)

    return preprocessed_img, text, corrected_text

# Streamlit interface
def main():
    st.markdown(
    f"""
    <style>
    body {{
        background-image: url('https://wallpaperaccess.com/full/6200122.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        height: 100vh;
        margin: 0;
        background-blend-mode: lighten; /* Experiment with different blend modes */
        opacity: 0.6; /* Adjust the opacity */
    }}
    </style>
    """,
    unsafe_allow_html=True
)
    st.title("OCR and Spell-checking App")

    # File uploader to upload an image file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the file to an array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Load the image
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.image(img, caption="Original Image", use_column_width=True)
        
        # Process the image
        preprocessed_img, extracted_text, corrected_text = process_image(img)
        
        # Display preprocessed image
        st.image(preprocessed_img, caption="Preprocessed Image", use_column_width=True)
        
        # Get image height and width
        h_img, w_img = preprocessed_img.shape
        
        # Get bounding boxes for each character
        boxes = pytesseract.image_to_boxes(preprocessed_img)
        
        # Draw bounding boxes and labels on the image
        for b in boxes.splitlines():
            b = b.split(' ')
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(img, (x, h_img - y), (w, h_img - h), (0, 0, 255), 2)
            cv2.putText(img, b[0], (x, h_img - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        
        # Display image with detected text and bounding boxes
        st.image(img, caption="Original Image with Detected Text", use_column_width=True)

# Run the app
if __name__ == '__main__':
    main()
