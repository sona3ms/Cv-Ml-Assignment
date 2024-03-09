from PIL import Image, ImageDraw, ImageFont
import os

# Directory to save generated images
output_dir = "training_data_1"
os.makedirs(output_dir, exist_ok=True)

# Text to be written on the image
text = "Hello, World!"

# Path to the directory containing font files
font_dir = r"C:\Users\HP\Desktop\Niqo Robotics\font"

# Size of the generated images
image_width = 200
image_height = 50

# Counter for the number of generated images
num_images_generated = 0

# Loop through each font file in the font directory
for font_file in os.listdir(font_dir):
    if font_file.endswith('.ttf'):
        font_path = os.path.join(font_dir, font_file)
        
        # Load the font
        font = ImageFont.truetype(font_path, size=24)
        
        # Create a new image with white background
        image = Image.new("RGB", (image_width, image_height), color="white")
        draw = ImageDraw.Draw(image)
        
        # Get the bounding box of the text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        
        # Calculate text width and height
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate text position
        text_x = (image_width - text_width) // 2
        text_y = (image_height - text_height) // 2
        
        # Write text on the image
        draw.text((text_x, text_y), text, fill="black", font=font)
        
        # Save the image
        font_name = os.path.splitext(font_file)[0]
        image_path = os.path.join(output_dir, f"{font_name}.png")
        image.save(image_path)
        
        # Increment the counter
        num_images_generated += 1

print("Total training images generated:", num_images_generated)

