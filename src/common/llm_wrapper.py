import os
import openai
from openai import Stream, ChatCompletion
import re

GPT3 = "gpt-3.5-turbo-16k-0613"
GPT4 = "gpt-4-turbo"
GPT4O_MINI = "gpt-4o-mini"
GPT4O = "gpt-4o"
# LLAMA3 = "meta-llama/Meta-Llama-3-8B-Instruct"
# LLAMA3 = "llama3.1:8b-instruct-fp16"
# LLAMA3 = "llama3.2:1b"
# LLAMA3 = "llama3.2:3b-instruct-q8_0"
LLAMA3 = "llama3.2:latest"
RWKV = "rwkv"
QWEN = "qwen2.5:7b-instruct-fp16"
QWEN_2_5_72B = "qwen2.5-72b-instruct"
INTERN = "internlm/internlm2.5:7b-chat-1m"
GEMMA2 = "gemma2:9b-instruct-fp16"
DEEPSEEKR1_8B = "deepseek-r1:8b-llama-distill-fp16"
DEEPSEEKR1_32B = "deepseek-r1:32b-qwen-distill-fp16"
QWQ_32B_LOCAL = "qwq:32b-fp16"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "chat_log.txt")
chat_log_path_with_history = os.path.join(CURRENT_DIR, "chat_log_with_history.txt")
openai_api_key = os.getenv("OPENAI_API_KEY", default="token-abc123")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", default="token-abc123")

class LLMWrapper:
    def __init__(self, temperature=0.0):
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
        self.rwkv_client = openai.OpenAI(
            base_url='http://localhost:8000',
            api_key="token-abc123",
        )
        self.history = {}
    
    def request_with_history(self, prompt, system_prompt=None, model_name=LLAMA3, history_id=None) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        if model_name == RWKV:
            client = self.rwkv_client
        elif model_name == GPT4 or model_name == GPT4O_MINI or model_name == GPT3 or model_name == GPT4O:
            client = self.gpt_client
        elif model_name == QWEN_2_5_72B:
            client = self.dashscope_client
        else:
            client = self.ollama_client
        
        
        
        history = self.history.get(history_id, [])

        
        # print(f"Requesting from {model_name}... use\n{prompt}")
        if system_prompt is not None:
            history.append({
                "role": "system", 
                "content": system_prompt
            })
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
        if len(history) > 5:
            history.pop(0)
        self.history[history_id] = history

    def clear_history(self, history_id):
        self.history[history_id] = []

    # TODO: 实现filter
    def request(self, prompt, system_prompt=None, model_name=LLAMA3, stream=False, multi_sentence=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        if model_name == RWKV:
            client = self.rwkv_client
        elif model_name == GPT4 or model_name == GPT4O_MINI or model_name == GPT3 or model_name == GPT4O:
            client = self.gpt_client
        elif model_name == QWEN_2_5_72B:
            client = self.dashscope_client
        else:
            client = self.ollama_client
        
        # print(f"Requesting from {model_name}... use\n{prompt}")
        if system_prompt is not None:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },{
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                stream=stream,
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                stream=stream,
            )

        # response = client.chat.completions.create(
        #     model=model_name,
        #     messages=[
        #         {
        #             "role": "user", 
        #             "content": [
        #                 {"type": "text", "text": prompt},
        #                 {
        #                     "type": "image_url",
        #                     "image_url": "data:image/png;base64,"
        #                 }
        #             ]
        #         }
        #     ],
        #     temperature=self.temperature,
        #     stream=stream,
        # )

        # save the message in a txt
        with open(chat_log_path, "a") as f:
            if system_prompt is not None:
                f.write(system_prompt + "\n\n--\n\n")
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