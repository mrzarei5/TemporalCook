
from openai import OpenAI
from dotenv import dotenv_values

from pyrate_limiter import Duration, Rate, Limiter, BucketFullException

from io import BytesIO
import base64
import time

rate = Rate(150, Duration.MINUTE)
limiter = Limiter(rate)



def make_request(client, model, text, image, tempt, top_p, seed):
    while True:
        try:
            limiter.try_acquire('api')  # 'api' is the identity or key
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                            },
                        },
                    ],
                }],
                temperature=tempt,
                top_p=top_p,
                #seed=seed,
            )
            return response
        except BucketFullException as e:
            # Sleep for the recommended time before retrying
            sleep_time = e.meta_info["remaining_time"]
            time.sleep(sleep_time)


class GeminiPro25():
    def __init__(self, model="gemini-2.5-pro", tempt=0, top_p=1, seed=42):
        config = dotenv_values(".env")        
        self.model = model
        self.client = OpenAI(api_key=config["GOOGLE_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
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
        response = make_request(self.client, self.model, text, base64_image, self.tempt, self.top_p, self.seed)

        return response.choices[0].message.content
