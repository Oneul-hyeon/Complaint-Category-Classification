import os
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from transformers import PreTrainedTokenizerFast
from src.GPT2 import convert_dialogue_to_features, GPT2ForSequenceClassification

def model_predict(dialogue) :
    input_ids, attention_masks, _ = convert_dialogue_to_features([dialogue], [0], MAX_LEN, tokenizer)
    input_ids = torch.LongTensor(input_ids).to(device)
    attention_masks = torch.LongTensor(attention_masks).to(device)
    out_ = model(input_ids, attention_masks)
    out = out_.detach().cpu().numpy()
    predict = np.argmax(out, axis = 1)[0]

    return mapping[encoderclass[predict]]

mapping = {"대중교통 안내" : "다산콜센터 : 대중교통 안내", "생활하수도 관련 문의" : "다산콜센터 : 생활하수도 관련 문의", "일반행정 문의" : "다산콜센터 : 일반행정 문의", "코로나19 관련 상담" : "다산콜센터 : 코로나19 관련 상담", "사고 및 보상 문의" : "금융/보험 : 사고 및 보상 문의", "상품 가입 및 해지" : "금융/보험 : 상품 가입 및 해지", "이체, 출금, 대출서비스" : "금융/보험 : 이체, 출금, 대출서비스", "잔고 및 거래내역" : "금융/보험 : 잔고 및 거래내역"}
'''
Data Preprocess
'''
os.chdir("..")
DATA_DIR = os.getcwd() + "/data/"
ENCODERCLASS_DIR = DATA_DIR + "encoderclass/"
CHECKPOINT_DIR = "model/checkpoint.pt"

# Setting Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token = '<s>', eos_token = '</s>', unk_token = '<unk>', mask_token = '<mask>', pad_token = '<pad>')

encoderclass = np.load(ENCODERCLASS_DIR + "ec.npy", allow_pickle = True)

# Parameter Setting
MAX_LEN = 350
batch_size = 8

model = GPT2ForSequenceClassification().to(device)
model_state_dict = torch.load(CHECKPOINT_DIR, map_location = device)
model.load_state_dict(model_state_dict)

model.eval()