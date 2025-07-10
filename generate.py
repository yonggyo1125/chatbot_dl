import os
import sys
import json
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='<s>', eos_token='</s>', pad_token='<pad>')
model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)

# 기준 경로
base_path = os.path.dirname(os.path.realpath(__file__))

if len(sys.argv) < 2:
    sys.exit(1)

model.load_weights(f"{base_path}/chatbot.weights.h5")

def chatbot(user_text):
    input_ids = tokenizer.encode(f"<s><usr>{user_text}<sys>")
    input_ids = tf.convert_to_tensor([input_ids])
    outputs = model.generate(input_ids, max_length=100, do_sample=True, top_k=100, top_p=0.9)
    sentence = tokenizer.decode(outputs[0].numpy().tolist())
    return sentence.split('<sys>')[1].replace('<pad>', '').replace('<unk>', '').replace("</s>", "")


message = sys.argv[1]
result = {"user": message, "system": chatbot(message)}
print(json.dumps(result, ensure_ascii=False))