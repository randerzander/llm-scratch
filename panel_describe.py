from PIL import Image 
import requests, base64, os
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 


url = "/home/dev/projects/llm-scratch/0.png"

processor = None
def im_prompt(url, prompt_txt):
    if processor is None:
        model_id = "microsoft/Phi-3-vision-128k-instruct" 

        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='eager') # use _attn_implementation='eager' to disable flash attention

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

    #image = Image.open(requests.get(url, stream=True).raw) 
    image = Image.open(url)

    messages = [{"role": "user", "content": f"<|image_1|>\n{prompt_txt}"}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": 500, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    return response

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


prompt_txt = "What is happening in this comic panel?"
for i in range(0, 5):
    fn = f"panel_{i}.png"
    resp = im_prompt(fn, prompt_txt)
    #resp = ai_playground(fn)
    print(f"{fn}: {resp}")
