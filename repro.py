import requests, base64
import os

invoke_url = "https://ai.api.nvidia.com/v1/vlm/community/llava16-34b"
stream = True

with open("/home/dev/projects/llm-scratch/panel_0.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

assert len(image_b64) < 180_000, \
  "To upload larger images, use the assets API (see docs)"

api_key = os.environ.get("NGC_API_KEY")
headers = {
  "Authorization": f"Bearer {api_key}",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "messages": [
    {
      "role": "user",
      "content": f'Describe the image. <img src="data:image/jpeg;base64,{image_b64}" />'
    }
  ],
  "max_tokens": 512,
  "temperature": 1.00,
  "top_p": 0.70,
  "stream": stream
}

response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())

def ai_playground(prompt, fn, model="llava16-34b"):
    invoke_url = f"https://ai.api.nvidia.com/v1/vlm/community/{model}"
    stream = True

    with open(fn, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    api_key = os.environ.get("NGC_API_KEY")
    headers = {
      "Authorization": f"Bearer {api_key}",
      "Accept": "text/event-stream" if stream else "application/json"
    }

    payload = {
      "messages": [
        {
          "role": "user",
          "content": f'{prompt} <img src="data:image/jpeg;base64,{image_b64}" />'
        }
      ],
      "max_tokens": 512,
      "temperature": 1.00,
      "top_p": 0.70,
      "stream": stream
    }

    response = requests.post(invoke_url, headers=headers, json=payload)

    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    else:
        return response.json()

ai_playground("Describe the image", "/home/dev/projects/llm-scratch/panel_0.png")
