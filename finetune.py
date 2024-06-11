# torchrun --nproc_per_node=4 finetune.py


import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification #DistilBertTokenizerFast, DistilBertForSequenceClassification, 
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

import os 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # for debbuging and syncronization

dataset = load_dataset("msc-smart-contract-audition/vulnerability-severity-classification")

model_name = 'distilbert-base-uncased'


class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    

def labels2index(labels):
    index = []
    for text in labels:
        if text == 'none':
            _ = 0
        elif text == 'low':
            _ = 1
        elif text == 'medium':
            _ = 2
        else:
            _ = 3
        index.append(_)

    return index 

def index2labels(index):
    labels = []
    for idx in index:
        if idx == 0:
            _ = 'none'
        elif idx == 1:
            _ = 'low'
        elif idx == 2:
            _ = 'medium'
        else:
            _ = 'high'
        labels.append(_)

    return labels


train_texts = dataset['train']['function']
train_labels = labels2index(dataset['train']['severity'])

test_texts = dataset['test']['function']
test_labels = labels2index(dataset['test']['severity'])


tokenizer = AutoTokenizer.from_pretrained(model_name)


train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


train_dataset = MyDataset(train_encodings, train_labels)
test_dataset = MyDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
model.to(torch.device('cuda'))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


trainer.train()

model_dir = "./results/models/cve"
tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir) 

print('Model saved successfully')