import os
import numpy as np
import openai
from openai import Stream, ChatCompletion
import re

LLAMA3V = "llama3.2-vision:11b-instruct-q8_0"
MINICPM = "minicpm-v:8b-2.6-fp16"
GPT4O_V = "gpt-4o"
INTERN_VL = "OpenGVLab/InternVL2_5-8B"
QWEN_VL_7B = "qwen2.5-vl-7b-instruct"
QWEN_VL_72B = "qwen2.5-vl-72b-instruct"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "chat_vlm_log_with_history.txt")
chat_log_path_with_history = os.path.join(CURRENT_DIR, "chat_vlm_log_with_history.txt")


openai_api_key = os.getenv("OPENAI_API_KEY", default="token-abc123")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", default="token-abc123")


import base64
from io import BytesIO
from PIL import Image
import cv2

def image_convert_color(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(im_rgb)
    return image

def image_to_base64(image, save_path=None):
    # image = image_convert_color(image)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    if save_path is not None:
        image.save(save_path, format="JPEG")
    img_str = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()
    return img_str

class VLMWrapper:
    def __init__(self, temperature=0.0):
        self.image_id = 0
        self.temperature = temperature
        self.ollama_client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="token-abc123",
        )
        self.gpt_client = openai.OpenAI(
            base_url='https://api.openai-proxy.org/v1',
            api_key=openai_api_key,
        )
        self.dashscope_client = openai.OpenAI(
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key=DASHSCOPE_API_KEY,
        )
        self.lmdeploy_client = openai.OpenAI(
            base_url='http://localhost:23333/v1',
            api_key="token-abc123",
        )
        self.history = {}
    
    def request_with_history(self, prompt, model_name=LLAMA3V, image=None, save_path=None, history_id=None) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        if model_name == GPT4O_V:
            client = self.gpt_client
        elif model_name == LLAMA3V or model_name == MINICPM:
            client = self.ollama_client
        elif model_name == QWEN_VL_7B or model_name == QWEN_VL_72B:
            client = self.dashscope_client
        else:
            client = self.lmdeploy_client
        
        # print(f"Requesting from {model_name}... use\n{prompt}")

        # prompt = "tell me about this image"

        history = self.history.get(history_id, [])

        if image is not None:
            self.image_id += 1
            if isinstance(image, Image.Image):
                if save_path is not None:
                    image = image_to_base64(image, save_path=save_path)
                else:
                    image = image_to_base64(image)
            else:
                if save_path is not None:
                    image = image_to_base64(cv2.imread(image), save_path=save_path)
                else:
                    image = image_to_base64(cv2.imread(image))

            history.append({
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        }
                    }
                ]
            })

        else:
            history.append({
                "role": "user", 
                "content": prompt
            })

        response = client.chat.completions.create(
            model=model_name,
            messages=history,
            temperature=self.temperature,
            stream=False,
        )

        # save the message in a txt
        with open(chat_log_path_with_history, "a") as f:
            f.write(str(history) + "\n---\n")
            f.write(response.model_dump_json(indent=2) + "\n---\n")

        return response.choices[0].message.content
    
    def update_history(self, history_id, content):
        history = self.history.get(history_id, [])
        history.append(content)
        self.history[history_id] = history

    def clear_history(self, history_id):
        self.history[history_id] = []

    def request(self, prompt, model_name=LLAMA3V, image=None, stream=False, multi_sentence=False, save_path=None) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        if model_name == GPT4O_V:
            client = self.gpt_client
        elif model_name == LLAMA3V or model_name == MINICPM:
            client = self.ollama_client
        elif model_name == QWEN_VL_7B:
            client = self.dashscope_client
        else:
            client = self.lmdeploy_client
        
        # print(f"Requesting from {model_name}... use\n{prompt}")

        # prompt = "tell me about this image"

        if image is not None:
            self.image_id += 1
            if isinstance(image, Image.Image):
                if save_path is not None:
                    image = image_to_base64(image, save_path=save_path)
                else:
                    image = image_to_base64(image)
            else:
                if save_path is not None:
                    image = image_to_base64(cv2.imread(image), save_path=save_path)
                else:
                    image = image_to_base64(cv2.imread(image))


        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature,
            stream=stream,
        )

        # save the message in a txt
        with open(chat_log_path, "a") as f:
            f.write(prompt + "\n---\n")
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response
        
        if multi_sentence:
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response.choices[0].message.content, re.DOTALL)
            return '\n'.join(code_blocks)
        else:
            return response.choices[0].message.content