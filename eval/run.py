import yaml
import os
import json
import torch
import time
from tqdm import tqdm
from get_action import *
from transfer import *
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
from eval_single import single_eval
from datetime import datetime
from test_loop import *

with open('config.yaml', 'r', encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file)


MODEL_PATH = config["MODEL_PATH"]
DATA_PATH = config["DATA_PATH"]
LOG_PATH = config["LOG_PATH"]
MODEL = config["MODEL"]
TEST_S = config["TEST_S"]
TEST_P = config["TEST_P"]
TEST_R = config["TEST_R"]
print(f"Model: {MODEL}")

if MODEL == "OS-Atlas" or MODEL == "UI-TARS":
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        trust_remote_code=True, 
        dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )   

elif MODEL == "GUI-owl" or MODEL == "UI-TARS-1.5" or MODEL == "Qwen2.5-VL" or MODEL == "UI-Venus":
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype="bfloat16", 
        device_map="auto",
        attn_implementation="flash_attention_2"
)

elif MODEL == "GELab" or MODEL == "Qwen3-VL" or MODEL == "MAI-UI" or MODEL == "UI-Venus-1.5":
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype="bfloat16", 
        device_map="auto",
        attn_implementation="flash_attention_2"
)

elif MODEL == "AgentCPM-GUI":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map="cuda:7",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

elif MODEL == "GLM-4.5V":
    model = None
else:
    print("Please enter the correct agent name in the MODEL section of config.yaml, or implement a new agent yourself.")


if MODEL == "GLM-4.5V":
    processor = None
else:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"logs_{MODEL}_{timestamp}"
folder_path = os.path.join(LOG_PATH, folder_name)
os.makedirs(folder_path, exist_ok=True)

if TEST_S:
    S_PATH = DATA_PATH + "S-subset"
    S_test_loop(model, processor, folder_path, DATA_PATH, tokenizer=None if MODEL!="AgentCPM-GUI" else tokenizer)

if TEST_P:
    P_PATH = DATA_PATH + "P-subset"
    P_test_loop(model, processor, folder_path, DATA_PATH, tokenizer=None if MODEL!="AgentCPM-GUI" else tokenizer)

if TEST_R:
    R_PATH = DATA_PATH + "R-subset"
    R_test_loop(model, processor, folder_path, DATA_PATH, tokenizer=None if MODEL!="AgentCPM-GUI" else tokenizer)

