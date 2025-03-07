# Do not need this anymore
import os
from PIL import Image

def process_images(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(directory, filename)
            with Image.open(filepath) as img:
                if img.size == (200, 75):
                    new_size = (int(img.width / 2.5), int(img.height / 2.5))
                    resized_img = img.resize(new_size)
                    resized_img.save(filepath)
                    print(f"Resized {filename} to {new_size}")
                else:
                    print(f"Skipped {filename}, size is {img.size}")

# Đặt đường dẫn đến thư mục chứa ảnh PNG
process_images("captcha_images")
