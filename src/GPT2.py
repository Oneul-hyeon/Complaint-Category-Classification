from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch import nn
from transformers import GPT2Model

def convert_dialogue_to_features(dialogues, labels, max_seq_len, tokenizer, pad_index=3):
    input_ids, attention_masks, data_labels = [], [], []

    for dialogue, label in zip(dialogues, labels):
        bos_token = [tokenizer.bos_token]
        eos_token = [tokenizer.eos_token]
        tokens = bos_token + tokenizer.tokenize(dialogue) + eos_token
        input_id = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_id)

        if len(input_id) < max_seq_len:
            pad_length = max_seq_len - len(input_id)
            pad = [pad_index] * pad_length
            input_id = input_id + pad
            attention_mask = attention_mask + [0] * pad_length
        else:
            input_id = input_id[:max_seq_len]
            attention_mask = attention_mask[:max_seq_len]

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        data_labels.append(label)

    return input_ids, attention_masks, data_labels

class GPT2Dataset(Dataset) :
    def __init__(self, input_ids, attention_masks, labels) :
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(attention_masks)
        self.labels = torch.LongTensor(labels)
        self.len = self.labels.shape[0]
    
    def __getitem__(self, index) :
        return self.input_ids[index], self.attention_masks[index], self.labels[index]
    
    def __len__(self) :
        return self.len

class GPT2ForSequenceClassification(nn.Module):
    def __init__(self, dr_rate = 0.2, num_labels=8):
        super(GPT2ForSequenceClassification, self).__init__()
        self.gpt = GPT2Model.from_pretrained('skt/kogpt2-base-v2')
        self.dropout = nn.Dropout(dr_rate)
        self.classifier = nn.Linear(self.gpt.config.hidden_size, num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)

    def forward(self, input_ids, attention_masks):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_masks)
        cls_token = outputs.last_hidden_state[:, -1, :]
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)
        return logits