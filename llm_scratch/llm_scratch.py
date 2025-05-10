import os, requests, json, time, sys, subprocess
from openai import OpenAI
import base64

def add_file(path):
    return open(path, "r").read()

def write_code(fn, llm_result):
    code = llm_result.replace("```python", "```")
    code = code.split("```")[1]
    with open(fn, "w") as f:
        f.write(code)

def perplexity(prompt: str, model="llama-3.1-sonar-large-128k-online"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["PERPLEXITY_API_KEY"], base_url="https://api.perplexity.ai")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": (prompt)
        },
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content

def haiku(prompt: str):
    return anthropic(prompt, model="claude-3-5-haiku-latest")

def opus(prompt: str):
    return anthropic(prompt, model="claude-3-opus-latest")

def anthropic(prompt: str, model="claude-3-5-sonnet-latest", max_tokens=2048):
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

def gemini(prompt: str, model="gemini-2.5-flash-preview-04-17"):
    from google import genai
    api_key = os.environ["GEMINI_API_KEY"]

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text


def make_messages(prompt):
    if type(prompt) == str:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        return messages
    else:
        return prompt

def openai(prompt, model="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI()

    return client.chat.completions.create(
        model=model,
        messages=make_messages(prompt)
    ).choices[0].message.content

def gpt_o1(prompt):
    return openai(prompt, model="o1-preview")

def gpt_o1_mini(prompt):
    return openai(prompt, model="o1-mini")

def gpt_4o(prompt, model="gpt-4o"):
    return openai(prompt, model=model)

def gpt_4o_mini(prompt, model="gpt-4o-mini"):
    return openai(prompt, model=model)

def meta_llama(prompt, model="nvidia/llama-3.1-nemotron-70b-instruct", max_tokens=2000, temperature=0.1, top_p=1):
  return aiplayground(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

def aiplayground(prompt, model, max_tokens=2000, temperature=0.1, top_p=1):
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

def tail_until_condition(file_path, target_line):
    with open(file_path, 'r') as file:
        file.seek(0, 2)  # Move to the end of the file
        while True:
            line = file.readline()
            if not line:
                time.sleep(0.1)  # Sleep briefly to avoid busy waiting
                continue
            #print(line.strip())  # Print the line without trailing newline
            if line.strip() == target_line:
                break

model_dir = "/home/dev/projects/models/"
pid = None
running_model = None
def start_server(model_path, mmproj_path=None):
    global pid
    global running_model
    if running_model == model_path:
        return
    else:
        stop_server()

    bin_path = "/home/dev/projects/llama.cpp/build/bin/llama-server"
    log_file = open("llama_server.log", "w")
    args = ["-m", model_path, "--n-gpu-layers", "1000", "--host", "0.0.0.0"]
    if mmproj_path is not None:
        args += ["--mmproj", mmproj_path]
    t0 = time.time()
    pid = subprocess.Popen([bin_path] + args, stdout=log_file, stderr=log_file)
    print(f"Started {model_path} with pid {pid.pid}")
    tail_until_condition("llama_server.log", "main: server is listening on http://0.0.0.0:8080 - starting the main loop")
    t1 = time.time()
    print(f"Server started in {t1-t0} seconds")
    running_model = model_path

def stop_server():
    global pid
    if pid is not None:
        pid.kill()
        pid.wait()
        pid = None
    subprocess.run(["pkill", "llama-server"])

def prep_content(prompt, prompt_template=None, image_path=None):
    if prompt_template is not None:
        prompt = prompt_template.replace("{prompt}", prompt)
    if image_path is not None:
        with open(image_path, "rb") as image_file:
            b64_image = base64.b64encode(image_file.read()).decode("utf-8")
        content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
        ]
    else:
        content = prompt
    return content

def llm(prompt, prompt_template=None, thinking=True, image_path=None):
    client = OpenAI(base_url="http://localhost:8080/v1")
    content = prep_content(prompt, prompt_template=prompt_template, image_path=image_path)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "type": "text",
            "content": content
        },
    ]
    if thinking:
        temperature = 0.6
        min_p = 0.0
        top_p = 0.95
        top_k = 20
    else:
        temperature = 0.7
        min_p = 0.0
        top_p = 0.8
        top_k = 20

    response = client.chat.completions.create(
        model="whatever",
        messages=messages,
        top_p=top_p,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content

def get_messages():
    global chat_messages
    return chat_messages

def empty_messages():
    global chat_messages
    chat_messages = None

chat_messages = None
def chat(prompt, thinking=True, image_path=None):
    client = OpenAI(base_url="http://localhost:8080/v1")
    global chat_messages
    content = prep_content(prompt, image_path=image_path)
    if chat_messages is None:
        chat_messages = [
            {
                "role": "system",
                "content": (
                    "You are an artificial intelligence assistant and you need to "
                    "engage in a helpful, detailed, polite conversation with a user."
                ),
            },
            {
                "role": "user",
                "content": content
            },
        ]
    else:
        chat_messages.append({"role": "user", "content": content})
    print(f"Messages: {len(chat_messages)}, chars: {len(str(chat_messages))}, tokens: {int(len(str(chat_messages))/4.5)}")
    #print(chat_messages)
    global running_model
    #print(running_model)

    if thinking:
        temperature = 0.6
        min_p = 0.0
        top_p = 0.95
        top_k = 20
    else:
        temperature = 0.7
        min_p = 0.0
        top_p = 0.8
        top_k = 20

    response = client.chat.completions.create(
        model="whatever",
        messages=chat_messages,
        top_p=top_p,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    chat_messages.append({"role": "system", "content": content})
    return content

def qwen(prompt=None):
    #model_path=f"{model_dir}Qwen3-14B-128K-UD-Q8_K_XL.gguf"
    model_path=f"{model_dir}Qwen3-30B-A3B-UD-Q4_K_XL.gguf"
    model_path=f"{model_dir}Qwen3-30B-A3B-Q8_0.gguf"
    start_server(model_path)
    if prompt is not None:
        return llm(prompt)

# qwen-14b 15638MiB
# 32b q4: 20226MiB
# 32b q5: 23386MiB
# 32b q6: 26742MiB
# 70b q5: 25000MiB 2x
def r1(prompt=None):
    model_path = f"{model_dir}DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf"
    model_path = f"{model_dir}DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
    model_path = f"{model_dir}DeepSeek-R1-Distill-Qwen-32B-Q5_K_M.gguf"
    model_path = f"{model_dir}DeepSeek-R1-Distill-Qwen-32B-Q6_K.gguf"
    model_path = f"{model_dir}DeepSeek-R1-Distill-Llama-70B-Q5_K_M.gguf"
    model_path = f"{model_dir}r1-1776-distill-llama-70b-Q5_K_M.gguf"
    start_server(model_path)
    if prompt is not None:
        return llm(prompt)

# 11542MiB
def gemma(prompt=None):
    model_path = f"{model_dir}gemma-3-27B-it-QAT-Q4_0.gguf"
    model_path = f"{model_dir}gemma-3-4b-it-UD-Q8_K_XL.gguf"

    start_server(model_path, mmproj_path=f"{model_dir}mmproj-F32.gguf")
    if prompt is not None:
        return llm(prompt)

# 15828MiB
def phi4(prompt=None):
    model_path = f"{model_dir}phi-4-Q8_0.gguf"
    start_server(model_path)
    if prompt is not None:
        return llm(prompt)
