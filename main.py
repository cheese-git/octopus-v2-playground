import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, GemmaForCausalLM
import torch
import time

# logging.basicConfig(level=logging.DEBUG)
# print(os.getenv("HF_ENDPOINT"))


def inference(input_text):
    start_time = time.time()
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = input_ids["input_ids"].shape[1]
    outputs = model.generate(
        input_ids=input_ids["input_ids"], max_length=1024, do_sample=False
    )
    generated_sequence = outputs[:, input_length:].tolist()
    res = tokenizer.decode(generated_sequence[0])
    end_time = time.time()
    return {"output": res, "latency": end_time - start_time}


model_id = "NexaAIDev/Octopus-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GemmaForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

input_text = "Take a selfie for me with front camera"
nexa_query = f"Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {input_text} \n\nResponse:"
start_time = time.time()
print("nexa model result:\n", json.dumps(inference(nexa_query), indent=4))
print("latency:", time.time() - start_time, " s")


{
    "output": " <nexa_0>('front')<nexa_end>\n\nFunction description: \ndef take_a_photo(camera):\n    \"\"\"\n    Captures a photo using the specified camera and resolution settings.\n\n    Parameters:\n    - camera (str): Specifies the camera to use. Can be 'front' or 'back'. The default is 'back'.\n\n    Returns:\n    - str: The string contains the file path of the captured photo if successful, or an error message if not. Example: '/storage/emulated/0/Pictures/MyApp/IMG_20240310_123456.jpg'\n    \"\"\"\n<eos>",
    "latency": 40.64596676826477,
}
