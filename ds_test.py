import requests
import json
import os

def r1(prompt):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    api_key = os.environ["NVIDIA_BUILD_API_KEY"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "messages": [
            {
                "content": prompt,
                "role": "user"
            }
        ],
        "model": "nvdev/deepseek-ai/deepseek-r1",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 4096,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()["choices"][0]["message"]["content"]

q = "hello, world"
response = r1(q)
print(response.json())
