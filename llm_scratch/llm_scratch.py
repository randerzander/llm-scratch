import os, requests, json, time, sys

def add_file(path):
    return open(path, "r").read()

def write_code(fn, llm_result):
    code = llm_result.replace("```python", "```")
    code = code.split("```")[1]
    with open(fn, "w") as f:
        f.write(code)

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

#def anthropic(prompt: str, model="claude-3-opus-20240229", max_tokens=2048):
def haiku(prompt: str):
    return anthropic(prompt, model="claude-3-haiku-20240307")

def opus(prompt: str):
    return anthropic(prompt, model="claude-3-opus-20240229")

def anthropic(prompt: str, model="claude-3-haiku-20240307", max_tokens=2048):
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content

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

def aiplayground(prompt, model="meta/llama3-70b", max_tokens=2000, temperature=0.1, top_p=1):
    from openai import OpenAI

    client = OpenAI(
      base_url = "https://integrate.api.nvidia.com/v1",
      api_key = os.environ["NGC_API_KEY"]
    )

    completion = client.chat.completions.create(
      model=model,
      messages=[{"role":"user","content":prompt}],
      temperature=temperature,
      top_p=top_p,
      max_tokens=max_tokens,
      stream=True
    )
    
    response = ""
    for chunk in completion:
      if chunk.choices[0].delta.content is not None:
          response += chunk.choices[0].delta.content
    return response

def llama3(prompt, model="meta/llama3-70b", max_tokens=2000):
    return aiplayground(prompt, model=model, max_tokens=max_tokens)

def mixtral(prompt, max_tokens=2048):
    return aiplayground(prompt, model="mistralai/mixtral-8x22b-instruct-v0.1", max_tokens=max_tokens)

llm = None
model_dir = "/home/dev/projects/models/"
#def llama_cpp(prompt, model_path=f"{model_dir}Llama-3-DARE-8B.Q8_0.gguf", max_tokens=2048):
#def llama_cpp(prompt, model_path=f"{model_dir}Meta-Llama-3-8B-Instruct.Q8_0.gguf", max_tokens=2048):
def llama_cpp(prompt, model_path=f"{model_dir}Hermes-2-Pro-Llama-3-8B-Q8_0.gguf", temperature=0.1, max_tokens=2049):
    from llama_cpp import Llama
    global llm
    if llm is None:
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096)
    #return llm(prompt, max_tokens=max_tokens, stop=["Q:", "\n"])
    # for vanilla 8b instruct
    #prompt_format = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n'
    prompt_format = """
<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
    """
    resp = llm(prompt_format.replace("{prompt}", prompt), max_tokens=max_tokens, temperature=temperature)
    return resp["choices"][0]["text"]
