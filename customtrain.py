import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import pytesseract

# Load the trained model
model = load_model('new_font_classification_model.h5')

# Directory containing the input images
input_images_dir = r"C:\Users\HP\Desktop\Niqo Robotics\test"
output_images_dir = r"C:\Users\HP\Desktop\Niqo Robotics\output_images"
os.makedirs(output_images_dir, exist_ok=True)

# Load the input images
input_images = []
for filename in os.listdir(input_images_dir):
    if filename.lower().endswith(('.png','.jpg','.jpeg')):
        image_path = os.path.join(input_images_dir, filename)
        try:
            input_image = Image.open(image_path)
            input_images.append((filename, input_image))
        except Exception as e:
            print(f"Error loading image {filename}: {e}")

# Function to extract words and their bounding boxes using OCR
def extract_words(image):
    # Perform OCR to detect words and their bounding boxes
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i]
        if word.strip() != '':
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            words.append({"text": word, "boundaryBox": {"x": x, "y": y, "width": w, "height": h}})
    return words

# Loop through each input image
for filename, input_image in input_images:
    try:
        # Extract words and their bounding boxes
        words = extract_words(input_image)

        # Predict font and confidence for each word
        detected_fonts = []
        for word_data in words:
            word = word_data["text"]
            bbox = word_data["boundaryBox"]

            # Crop the region defined by the bounding box
            cropped_image = input_image.crop((bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]))

            # Preprocess the cropped image (resize, convert to numpy array, normalize)
            cropped_image = cropped_image.resize((200, 50))  # Resize to match input size of the model
            cropped_image_array = np.array(cropped_image) / 255.0  # Normalize pixel values
            cropped_image_array = np.expand_dims(cropped_image_array, axis=0)  # Add batch dimension

            # Classify the font using the trained model
            font_class_probabilities = model.predict(cropped_image_array)
            predicted_font_class = np.argmax(font_class_probabilities)
            confidence = float(np.max(font_class_probabilities))

            # Decode the predicted font class index to its actual font label
            font_labels = ['Roboto', 'DancingScript', 'NotoSans', 'OpenSans', 'Oswald', 'PatuaOne', 'PTSerif', 'Arimo', 'Ubuntu']
            predicted_font_label = font_labels[predicted_font_class]

            # Add detected font to the list
            detected_fonts.append({
                "word": word,
                "boundaryBox": bbox,
                "font": predicted_font_label,
                "confidence": confidence
            })

        # Print the detected fonts
        output = {"detectedFonts": detected_fonts}
        print(output)

        # Save the output image with bounding boxes around words and font labels
        draw = ImageDraw.Draw(input_image)
        for word_data in detected_fonts:
            bbox = word_data["boundaryBox"]
            draw.rectangle([bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]], outline="red")
            draw.text((bbox["x"], bbox["y"] - 15), f"{word_data['font']} - Confidence: {word_data['confidence']:.2f}", fill="red")

        # Save the image with bounding boxes and font labels
        output_image_path = os.path.join(output_images_dir, filename)
        input_image.save(output_image_path)
        print(f"Output image saved to: {output_image_path}")
    except Exception as e:
        print(f"Error processing image {filename}: {e}")
