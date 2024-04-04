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
