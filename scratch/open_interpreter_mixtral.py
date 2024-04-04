import os
from openai import OpenAI
from interpreter import interpreter

interpreter.offline = True # Disables online features like Open Procedures
interpreter.llm.model = "playground_mixtral_8x7b" # Tells OI to send messages in OpenAI's format
interpreter.llm.api_key = os.environ["NGC_API_KEY"]
interpreter.llm.api_base = "https://integrate.api.nvidia.com/v1"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NGC_API_KEY"],
)


interpreter.chat()
