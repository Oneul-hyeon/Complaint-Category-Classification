import os
import warnings
warnings.filterwarnings("ignore")
from ast import literal_eval
from transformers import PreTrainedTokenizerFast
from sklearn.model_selection import train_test_split
from pytorchtools import EarlyStopping
from preprocessing import label_encoding, split_dialogue
from tqdm import tqdm
from GPT2 import convert_dialogue_to_features, GPT2Dataset, GPT2ForSequenceClassification

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim

import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
gamma_neg, gamma_pos : 음수와 양수 예측에 대한 가중치 조정 파라미터
clip : 음수 손실을 클리핑하여 너무 작은 값이 되지 않도록 방지
reduction : 손실 값을 어떻게 축약할지 결정
'''

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross Entropy Loss
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
    
def calc_accuracy(X, Y) :
    _, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc

os.chdir("..")
df = pd.read_csv("data/df/train_df.csv")

df["dialogue"] = df.dialogue.apply(lambda x : literal_eval(x))
df["split_dialogue"] = df.dialogue.apply(lambda x : split_dialogue(x))
df["label"] = label_encoding(df)

train_df, valid_df = train_test_split(df, test_size = 0.1, stratify=df["label"], random_state = 42)

tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token = '<s>', eos_token = '</s>', unk_token = '<unk>', mask_token = '<mask>', pad_token = '<pad>')

MAX_LEN = 350
batch_size = 8
epoch = 1000
patience = 5
lr = 5e-5

train_input_ids, train_attention_masks, train_labels = convert_dialogue_to_features(train_df["split_dialogue"], train_df["label"], MAX_LEN, tokenizer)
valid_input_ids, valid_attention_masks, valid_labels = convert_dialogue_to_features(valid_df["split_dialogue"], valid_df["label"], MAX_LEN, tokenizer)

class_counts = np.bincount(train_df["label"])
class_weights = 1. / class_counts
samples_weights = class_weights[train_df["label"]]

sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

train_dataset = GPT2Dataset(train_input_ids, train_attention_masks, train_labels)
valid_dataset = GPT2Dataset(valid_input_ids, valid_attention_masks, valid_labels)

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 2, sampler = sampler)
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, num_workers = 2, shuffle = False)

model = GPT2ForSequenceClassification().to(device)

early_stopping = EarlyStopping(patience = patience, verbose = True, path = "model/checkpoint.pt")

# loss_fn = CrossEntropyLoss()
loss_fn = FocalLoss()

optimizer = optim.Adam(model.parameters(), lr = lr)

for e in range(1, epoch+1) :
    train_acc, val_acc = 0.0, 0.0
    train_loss, val_loss = 0.0, 0.0
    model.train()
    for batch_id, (input_ids, attention_masks, labels) in enumerate(tqdm(train_dataloader)) :
        optimizer.zero_grad()
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        out = model(input_ids, attention_masks)
        
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += calc_accuracy(out, labels)

    print("epoch {} train_acc {}".format(e, train_acc / (batch_id + 1)))
    
    model.eval()
    for batch_id, (input_ids, attention_masks, labels) in enumerate(tqdm(valid_dataloader)) :
        with torch.no_grad() :
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            out = model(input_ids, attention_masks)

            loss = loss_fn(out, labels)
            val_loss += loss.item()
            val_acc += calc_accuracy(out, labels)
    print("epoch {} val_acc {} val_loss {}".format(e, val_acc / (batch_id + 1), val_loss / len(valid_dataloader)))
    early_stopping(val_loss / len(valid_dataloader), model)
    if early_stopping.early_stop : break
print("End Training...")