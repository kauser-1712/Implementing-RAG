import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")
print(load_env())
