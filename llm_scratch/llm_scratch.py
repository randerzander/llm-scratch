import os, requests, json, time, sys

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

llm = None
model_dir = "/home/dev/projects/models/"
prompt_format_mistral_nemo= """
<s>[INST]{prompt}[/INST]
"""
prompt_format_gemma = """
<bos><start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
"""
prompt_format_llama = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
#def llama_cpp(prompt, model_path=f"{model_dir}Mistral-Nemo-Instruct-2407.Q8_0.gguf", temperature=0.1, max_tokens=2049):
def llama_cpp(prompt, model_path=f"{model_dir}qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf", temperature=0.1, max_tokens=2049):
    from llama_cpp import Llama
    global llm
    if llm is None:
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=15000, verbose=False)
    #resp = llm(prompt_format_mistral_nemo.replace("{prompt}", prompt), max_tokens=max_tokens, temperature=temperature, )
    stop_words = ["\n\n", "Human:", "Assistant:", "<|im_end|>"]
    resp = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop_words)
    return resp["choices"][0]["text"]
