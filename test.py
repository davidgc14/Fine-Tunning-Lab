from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
import torch
import torch.nn.functional as F
from datasets import load_dataset
import random
import numpy as np

test_size = 437 # all samples

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


dataset = load_dataset("msc-smart-contract-audition/vulnerability-severity-classification")
test_dataset = dataset['test']

values = random.sample(range(len(test_dataset['severity'])), test_size)

test_texts = [test_dataset['function'][index] for index in values]
test_labels = [test_dataset['severity'][index] for index in values]



model_name = 'results/models/cve'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

batch = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')




with torch.no_grad():
    outputs = model(**batch)
    prediction = F.softmax(outputs.logits, dim=1)
    labels = torch.argmax(prediction, dim=1).tolist()
    label_name = index2labels(labels)

# print(label_name)

# print(test_labels)



# evaluate

accuracy = sum(np.array(label_name) == np.array(test_labels)) / test_size

print('Accuracy:', round(accuracy, 4))