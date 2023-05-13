from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
# A optimizer
from transformers import AdamW

# Prepare dataset
# Load pretrained Tokenizer call it with dataset -> encoding
# Build PyTorch Dataset with encodings
# Load pretrained Model
# Load Trainer and train it or use native PyTorch training pipeline

model_name = 'distilbert-base-uncased'

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    
    for label_dir in ['pos', 'neg']:
        for text_file in (split_dir / label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == 'neg' else 1)
    
    return texts, labels

# Large Movie Review Datset
# https://ai.stanford.edu/~amaas/data/sentiment/
train_texts, train_labels = read_imdb_split('train')
test_texts, test_labels = read_imdb_split('test')

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size = 0.2)

def IMDbDataset(Dataset):
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        
        return item
    
    def __len__(self):
        return len(self.labels)
    
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Ensure that all of our sequences are padded to the same length and are truncated to be no longer than model's
# maximum input length. This will allow us to feed batches of sequences into the model at the same time.
train_encodings = tokenizer(train_texts, truncation = True, padding = True)
val_encodings = tokenizer(val_texts, truncation = True, padding = True)
test_encodings = tokenizer(test_texts, truncation = True, padding = True)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

training_args = TrainingArguments(output_dir = './results', num_train_epochs = 2, per_device_train_batch_size = 16, per_device_eval_batch_size = 64, warmup_steps = 500, learning_rate = 5e-5, weight_decay = 0.01, logging_dir = './logs', logging_steps = 10)

model = DistilBertForSequenceClassification.from_pretrained(model_name)

trainer = Trainer(model = model, args = training_args, train_dataset = train_dataset, eval_dataset = val_dataset)

trainer.train()

# If you want to do this manually
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DistilBertForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
optim = AdamW(model.parameters(), lr = 5e-5)
num_train_epochs = 2
for epoch in range(num_train_epochs):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()