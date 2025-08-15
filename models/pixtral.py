
from dotenv import dotenv_values

from io import BytesIO
import base64
from mistralai import Mistral


class PixtralLarge():
    def __init__(self, model="pixtral-large-latest",tempt=0, top_p=1, seed=42):
        config = dotenv_values(".env")        
        self.model = model
        self.client = Mistral(api_key=config["MISTRALAI_API_KEY"])
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
        response = self.client.chat.complete(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    },
                ],
            }],
            temperature=self.tempt,
            top_p=self.top_p
            #seed=self.seed,
        )

        return response.choices[0].message.content
