import os
from openai import OpenAI
from interpreter import interpreter

interpreter.offline = True # Disables online features like Open Procedures
#interpreter.llm.model = "sonar-small-chat" # Tells OI to send messages in OpenAI's format
interpreter.llm.model = "sonar-small-online" # Tells OI to send messages in OpenAI's format
interpreter.llm.api_key = os.environ["PERPLEXITY_API_KEY"]
interpreter.llm.api_base = "https://api.perplexity.ai"

interpreter.chat()
