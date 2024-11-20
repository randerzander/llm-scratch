import time, os
import requests, base64

tokenizer, model, image_processor, context_len = None, None, None, None

det_processor, det_model = None, None
rec_model, rec_processor = None, None

def extract_text(image_fn):
    from PIL import Image
    from surya.ocr import run_ocr
    from surya.model.detection import segformer
    from surya.model.recognition.model import load_model
    from surya.model.recognition.processor import load_processor

    langs = ["en"] # Replace with your languages

    global det_processor, det_model, rec_model, rec_processor
    if det_processor is None:
        det_processor, det_model = segformer.load_processor(), segformer.load_model()
        rec_model, rec_processor = load_model(), load_processor()

    t0 = time.time()
    image = Image.open(image_fn)
    predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
    pred = predictions[0]
    text = [line.text for line in pred.text_lines]
    t1 = time.time()
    print(t1-t0)
    return text

def yolo(path):
    from inference import get_model
    model = get_model(model_id="yolov8n-640")
    return model.infer(path)

def kosmos(image_file, prompt):
  invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2"

  with open(image_file, "rb") as f:
      image_b64 = base64.b64encode(f.read()).decode()

  assert len(image_b64) < 180_000, \
    "To upload larger images, use the assets API (see docs)"

  headers = {
    "Authorization": f"Bearer {os.environ['NGC_API_KEY']}",
    "Accept": "application/json"
  }

  payload = {
    "messages": [
      {
        "role": "user",
        "content": f'{prompt} <img src="data:image/png;base64,{image_b64}" />'
      }
    ],
    "max_tokens": 1024,
    "temperature": 0.20,
    "top_p": 0.20
  }

  response = requests.post(invoke_url, headers=headers, json=payload)
  return response.json()


def image_prompt(image_file, prompt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image

    model_id = "vikhyatk/moondream2"
    revision = "2024-03-05"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    t0 = time.time()
    image = Image.open(image_file)
    enc_image = model.encode_image(image)
    resp = model.answer_question(enc_image, prompt, tokenizer)
    t1 = time.time()
    print(t1-t0)
    return resp

def llava_image_prompt(image_file, prompt):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.eval.run_llava import eval_model

    global tokenizer
    global model
    global image_processor
    global context_len

    model_path = "liuhaotian/llava-v1.5-7b"
    ##model_path = "liuhaotian/llava-v1.5-13b"
    #model_path = "TheBloke/llava-v1.5-13B-AWQ"

    if tokenizer is None:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )

    t0 = time.time()
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    result = eval_model(args)
    t1 = time.time()
    print(t1-t0)
    return result

flux_pipe = None
def flux_drop():
    import torch, gc
    global flux_pipe
    del flux_pipe
    flux_pipe = None
    torch.cuda.empty_cache()
    gc.collect()

def flux(prompt, fn, num_inference_steps=4):
    from diffusers import FluxPipeline
    import torch
    global flux_pipe
    if flux_pipe is None:
        flux_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        flux_pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    t0 = time.time()
    image = flux_pipe(
        prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=num_inference_steps,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image.save(fn)
    t1 = time.time()
    print(f"Time taken: {t1-t0:.2f}s")

eagle_tokenizer, eagle_model, eagle_image_processor, eagle_context_len = None, None, None, None
def eagle_drop():
    import torch, gc
    global eagle_tokenizer, eagle_model, eagle_image_processor, eagle_context_len
    del eagle_tokenizer
    eagle_model.cpu()
    del eagle_model
    del eagle_image_processor
    del eagle_context_len
    eagle_tokenizer, eagle_model, eagle_image_processor, eagle_context_len = None, None, None, None
    torch.cuda.empty_cache()
    gc.collect()

def eagle(image_path, input_prompt):
    t0 = time.time()
    model_path = "NVEagle/Eagle-X5-7B"
    conv_mode = "vicuna_v1"

    from eagle import conversation as conversation_lib
    from eagle.constants import DEFAULT_IMAGE_TOKEN
    from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from eagle.conversation import conv_templates, SeparatorStyle
    from eagle.model.builder import load_pretrained_model
    from eagle.utils import disable_torch_init
    from eagle.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images, KeywordsStoppingCriteria
    from PIL import Image
    import argparse
    from transformers import TextIteratorStreamer
    from threading import Thread
    import torch

    model_name = get_model_name_from_path(model_path)
    global eagle_tokenizer, eagle_model, eagle_image_processor, eagle_context_len
    if eagle_tokenizer is None:
        eagle_tokenizer, eagle_model, eagle_image_processor, eagle_context_len = load_pretrained_model(model_path,None,model_name,False,False)
    if eagle_model.config.mm_use_im_start_end:
        input_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + input_prompt
    else:
        input_prompt = DEFAULT_IMAGE_TOKEN + '\n' + input_prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], eagle_image_processor, eagle_model.config)[0]
    input_ids = tokenizer_image_token(prompt, eagle_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    input_ids = input_ids.to(device='cuda', non_blocking=True)
    image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

    with torch.inference_mode():
        output_ids = eagle_model.generate(
            input_ids.unsqueeze(0),
            images=image_tensor.unsqueeze(0),
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.2,
            top_p=0.5,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True)

    outputs = eagle_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    t1 = time.time()
    print(f"Time taken: {t1-t0:.2f}s")
    return outputs
