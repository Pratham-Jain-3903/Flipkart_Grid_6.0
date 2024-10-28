import os
import time
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import random

class FreshnessDetector:
    def __init__(self):
        # Load the model and processor for Qwen2-VL
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,  # Mixed precision for efficiency
            device_map="auto"  # Automatically place on GPU if available
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        self.image_folder = r"C:\Users\prath\PycharmProjects\Flipkart_Robotics\Real-Time-Object-Detection-With-OpenCV\bottle_images"

    def process_image(self, image_path):
        """Load, resize, and process image for model input."""
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((512, 512))
        return image

    def generate_response(self, image, text_query):
        """Generate a response for the given image and text query."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_query}
                ]
            }
        ]

        # Apply chat template for the text prompt
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Prepare inputs
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        # Generate output
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024)

        # Decode the generated output text
        output_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return output_text[0]

    def check_freshness(self, category):
        """Check freshness by picking images from a folder and using Qwen-VL to analyze them."""
        if category in ["Fruits and Vegetables"]:
            image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

            if image_files:
                image_path = os.path.join(self.image_folder, image_files[0])  # Picking the first available image
                image = self.process_image(image_path)
                output_text = self.generate_response(image, "Give the freshness of the fruit on a scale of 1 to 100% direct answer one word eg:30")

                print(f"Detected Freshness: {output_text}")
                return output_text
            else:
                print("No image files found.")
                return "No image available"
        return 100

# Example of usage:
detector = FreshnessDetector()
detector.check_freshness("Fruits and Vegetables")
