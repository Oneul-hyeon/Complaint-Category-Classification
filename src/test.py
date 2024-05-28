import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from ast import literal_eval
from transformers import PreTrainedTokenizerFast
from preprocessing import split_dialogue
from tqdm import tqdm
from GPT2 import convert_dialogue_to_features, GPT2Dataset, GPT2ForSequenceClassification

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir("..")
CHECKPOINT_DIR = f"{os.getcwd()}/model/checkpoint.pt"
test_df = pd.read_csv("data/df/test_df.csv")

test_df["dialogue"] = test_df.dialogue.apply(lambda x : literal_eval(x))
test_df["split_dialogue"] = test_df.dialogue.apply(lambda x : split_dialogue(x))

label_encoder = np.load(f"{os.getcwd()}/data/encoderclass/ec.npy", allow_pickle = True)
encoder = {category : idx for idx, category in enumerate(label_encoder)}

test_df["label"] = test_df.category.apply(lambda x : encoder[x])

tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token = '<s>', eos_token = '</s>', unk_token = '<unk>', mask_token = '<mask>', pad_token = '<pad>')

MAX_LEN = 350
batch_size = 8
epoch = 1000
patience = 5
lr = 5e-5

test_input_ids, test_attention_masks, test_labels = convert_dialogue_to_features(test_df["split_dialogue"], test_df["label"], MAX_LEN, tokenizer)

test_dataset = GPT2Dataset(test_input_ids, test_attention_masks, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = 2, shuffle = False)

model = GPT2ForSequenceClassification().to(device)
model_state_dict = torch.load(CHECKPOINT_DIR, map_location = device)
model.load_state_dict(model_state_dict)

model.eval()
output = []
    
for batch_id, (input_ids, attention_masks, labels) in enumerate(tqdm(test_dataloader)) :
    with torch.no_grad() :
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        out = model(input_ids, attention_masks)

        for logits in out :
            output.append(logits.detach().cpu().numpy())
    
predict = np.argmax(output, axis = 1)

print(classification_report(test_df["label"], predict))