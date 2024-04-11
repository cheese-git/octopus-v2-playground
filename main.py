from llm import inference
from fastapi import FastAPI
from pydantic import BaseModel
# logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

class Input(BaseModel):
    input_text: str

@app.post("/query")
def query(input: Input):
    return inference(input.input_text)
