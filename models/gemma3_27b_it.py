
from openai import OpenAI
from dotenv import dotenv_values

from io import BytesIO
import base64


class Gemma3_27b_it():
    def __init__(self, model="google/gemma-3-27b-it-fast", tempt=0, top_p=1, seed=42):
        config = dotenv_values(".env")        
        self.model = model
        self.client = OpenAI(api_key=config["HF_API_KEY"], base_url="https://router.huggingface.co/nebius/v1")
        self.tempt = tempt
        self.top_p = top_p
        self.seed = seed

    def ask(self, image, text, ins_prompt = None):
        if ins_prompt:
            text = text + ' ' + ins_prompt 
        buffer = BytesIO()

        # Save the image to the buffer (in-memory, not disk)
        image.save(buffer, format="JPEG")  # or "PNG"

        # Get the bytes and encode to base64
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }],
            temperature=self.tempt,
            top_p=self.top_p,
            seed=self.seed,
        )

        return response.choices[0].message.content
