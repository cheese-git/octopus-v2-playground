import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = ".huggingface_cache"

from transformers import (
    AutoTokenizer,
    GemmaForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)
import torch
import time

model_id = "NexaAIDev/Octopus-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GemmaForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

class MyStoppingCriteria(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        keyword = tokenizer.decode(input_ids[0][-1])
        return keyword in ["<nexa_end>"]

def inference(input_text):
    start_time = time.time()
    nexa_query = f"Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {input_text} \n\nResponse:"
    input_ids = tokenizer(nexa_query, return_tensors="pt").to(model.device)
    input_length = input_ids["input_ids"].shape[1]
    output = model.generate(input_ids["input_ids"], max_length=1024, do_sample=False, stopping_criteria=StoppingCriteriaList([MyStoppingCriteria()]))
    generated_sequence = output[:, input_length:].tolist()

    return {
      "output": tokenizer.decode(generated_sequence[0]),
      "latency": time.time() - start_time
    }


if __name__ == "__main__":
    input_text = "Write an email to John, tell him the job is done."

    print(inference(input_text))
