import os
from threading import Thread

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, GemmaForCausalLM, TextIteratorStreamer
import torch
import time

# logging.basicConfig(level=logging.DEBUG)
# print(os.getenv("HF_ENDPOINT"))


def inference(input_text):
    start_time = time.time()
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    thread = Thread(target=model.generate, kwargs={"input_ids": input_ids["input_ids"], "max_length":1024, "do_sample":False, "streamer" : streamer})
    thread.start()
    
    text = ''
    
    for new_text in streamer:
        text += new_text
        if('<nexa_end>' in new_text):
            end_time = time.time()
            print(text)
            print("latency:", end_time - start_time, " s")
            break
        
    end_time = time.time()


model_id = "NexaAIDev/Octopus-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GemmaForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

input_text = "Take a selfie for me with front camera"
nexa_query = f"Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {input_text} \n\nResponse:"

inference(nexa_query)