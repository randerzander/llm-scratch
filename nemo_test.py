from nemollm.api import NemoLLM 
import os

conn = NemoLLM()

# advanced initialization
API_HOST = "https://api.llm.ngc.nvidia.com/v1"
#API_KEY = os.environ.get("OLD_NGC_API_KEY")
API_KEY = "dWJhdDlnYm1jYXM1a3Y0ZGFoNnF2NTRwbmQ6OGZhZjRlNTUtNjFjOC00MjgyLTliZDUtM2U0MDIzY2ZhNzQ4"
ORG_ID = "bwbg3fjn7she"

conn = NemoLLM(
    api_host=API_HOST,
    api_key=API_KEY,
    org_id=ORG_ID   
)

res = conn.list_models()
names = [model["name"] for model in res["models"]]
print(names)
