from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

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
train_texts, train_labels = read_imdb_split('aclImdb/train')
test_texts, test_labels = read_imdb_split('aclImdb/test')

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size = 0.2)

def IMDbDataset(Dataset):
    
    def __init(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {ley: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        
        return item
    
    def __len__(self):
        return len(self.labels)
    
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Ensure that all of our sequences are padded to the same length and are truncated to be no longer than model's
# maximum input length. This will allow us to feed batches of sequences into the model at the same time.