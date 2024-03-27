import os, requests, json, time, sys

def add_file(path):
    return open(path, "r").read()

def write_code(fn, llm_result):
    code = llm_result.replace("```python", "```")
    code = code.split("```")[1]
    with open(fn, "w") as f:
        f.write(code)

#def perplexity(prompt: str, model="pplx-7b-online"):
def perplexity(prompt: str, model="sonar-medium-online"):
    url = "https://api.perplexity.ai/chat/completions"
    perplexity_api_key = os.environ["PERPLEXITY_API_KEY"]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        'temperature': 1
    }
    headers = {
        "Authorization": f"Bearer {perplexity_api_key}",
        "accept": "application/json",
        "content-type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()['choices'][0]['message']['content']

def make_messages(prompt):
    if type(prompt) == str:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    else:
        return prompt

def gpt4(prompt, model="gpt-4-0125-preview"):
    from openai import OpenAI
    client = OpenAI()

    return client.chat.completions.create(
        model=model,
        messages=make_messages(prompt)
    ).choices[0].message.content

def smaug(prompt):
    invoke_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/008cff6d-4f4c-4514-b61e-bcfad6ba52a7"
    fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"

    ngc_api_key = os.environ["NGC_API_KEY"]
    headers = {
        "Authorization": f"Bearer {ngc_api_key}",
        "Accept": "application/json",
    }

    payload = {
      "messages": make_messages(prompt),
      "temperature": 0.2,
      "top_p": 0.7,
      "max_tokens": 1024,
      "seed": 42,
      "bad": None,
      "stop": None,
      "stream": False
    }

    session = requests.Session()

    response = session.post(invoke_url, headers=headers, json=payload)

    while response.status_code == 202:
        request_id = response.headers.get("NVCF-REQID")
        fetch_url = fetch_url_format + request_id
        response = session.get(fetch_url, headers=headers)

    response.raise_for_status()
    response_body = response.json()
    return response_body["choices"][0]["message"]["content"]

def mixtral(prompt, temperature=0.1, top_p=1, max_tokens=4096):
    from openai import OpenAI
    client = OpenAI(
      base_url = "https://integrate.api.nvidia.com/v1",
      api_key = os.environ["NGC_API_KEY"]
    )
    completion = client.chat.completions.create(
      model="mistralai/mixtral-8x7b-instruct-v0.1",
      messages=[{"role":"user","content":prompt}],
      temperature=temperature,
      top_p=top_p,
      max_tokens=max_tokens,
      stream=False
    )
    return completion.choices[0].message.content


llm = None
model_dir = "/home/dev/projects/models/"
#def llama_cpp(prompt, model_path=f"{model_dir}mistral-7b-instruct-v0.2.Q8_0.gguf", max_tokens=2048):
def llama_cpp(prompt, model_path=f"{model_dir}nous-hermes-2-mixtral-8x7b-dpo.Q4_K_M.gguf", max_tokens=2048):
    from llama_cpp import Llama
    global llm
    if llm is None:
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096)
    #return llm(prompt, max_tokens=max_tokens, stop=["Q:", "\n"])
    resp = llm(prompt, max_tokens=max_tokens)
    return resp["choices"][0]["text"]
