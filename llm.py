import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from threading import Thread
from transformers import AutoTokenizer, GemmaForCausalLM, TextIteratorStreamer, StoppingCriteria
import torch
import time

model_id = "NexaAIDev/Octopus-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GemmaForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

def inference(input_text):
    start_time = time.time()
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    thread = Thread(target=model.generate, kwargs={"input_ids": input_ids["input_ids"], "max_length":1024, "do_sample":False, "streamer" : streamer})
    thread.start()
    
    text = ''
    
    for new_text in streamer:
        text += new_text
        if('<nexa_end>' in new_text):
            end_time = time.time()
            return {"token": text, "latency": end_time - start_time}
    


if __name__ == '__main__':
    input_text = "Write an email to John, tell him the job is done."
    nexa_query = f"Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {input_text} \n\nResponse:"

    print(inference(nexa_query))