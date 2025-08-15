
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch


class LLAVA():
    def __init__(self, model_path="llava-hf/llava-v1.6-mistral-7b-hf", temperature = 0.001):
        
        
        self.processor = LlavaNextProcessor.from_pretrained(model_path)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        self.model.to("cuda:0")

        self.temperature = temperature
    def ask(self, image, text, ins_prompt = None):
        if ins_prompt:
            text = text + ' ' + ins_prompt 
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=512, temperature = self.temperature)

        response = self.processor.decode(output[0], skip_special_tokens=True)
        return response.split('[/INST]')[1]
