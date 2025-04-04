import llama_cpp
import instructor
from pydantic import BaseModel
from llm_scratch import get_plain_text_from_url

# Initialize the Llama model
llm = llama_cpp.Llama(
    model_path="/home/dev/projects/models/Hermes-3-Llama-3.1-8B.Q8_0.gguf",
    n_gpu_layers=-1,
    chat_format="chatml",
    n_ctx=15000,
    verbose=True
)

# Patch the create function with instructor
create = instructor.patch(
    create=llm.create_chat_completion_openai_v1,
    mode=instructor.Mode.JSON_SCHEMA
)

# Define your Pydantic model
class Template(BaseModel):
    name: str
    memory: str
    processor: str
    OS: str
    screen_size: str

# Use the patched function to get structured output
text = get_plain_text_from_url("https://liliputing.com/genbook-rk3588-is-modular-linux-laptop-with-an-upgradeable-design-crowdfunding/")
extracts = create(
    response_model=Template,
    messages=[
        {"role": "user", "content": text}
    ]
)

print(extracts)
