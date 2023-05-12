from transformers import pipeline
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 2 methods: Using the pipeline or tokenizing

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Default model is the model passed as a parameter
classifier = pipeline('sentiment-analysis', model = model, tokenizer = tokenizer)
results = classifier(['We are very happy to show you the 🤗 Transformers library.', 'We hope you do not hate it.'])

for result in results:
    print(result)

# 2 different methods shown below that get to the same results
tokens = tokenizer.tokenize('We are very happy to show you the 🤗 Transformers library.')
token_ids = tokenizer.convert_tokens_to_ids(tokens = tokens)
input_ids = tokenizer('We are very happy to show you the 🤗 Transformers library.')

print(f'    Tokens: {tokens}')
# Numerical representation that our model can understand
print(f'Token IDs: {token_ids}')
# 101: Beginning of string
# 102: End of string
print(f'Input IDs: {input_ids}')

X_train = ['We are very happy to show you the 🤗 Transformers library.', 'We hope you do not hate it.']
# pt: PyTorch
batch = tokenizer(X_train, padding = True, truncation = True, max_length = 512, return_tensors = 'pt')
print('Batch:', batch)

# Gradient
with torch.no_grad():
    # Unpack the dictionary values with those *
    # Labels parameter was not necessary unless you want to know the loss in the classifier output
    outputs = model(**batch, labels = torch.tensor([1, 1]))
    print('Outputs:', outputs)
    predictions = F.softmax(outputs.logits, dim = 1)
    print('Predictions:', predictions)
    labels = torch.argmax(predictions, dim = 1)
    print('Labels:', labels)
    # List comprehension
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print('Labels:', labels)

# Assuming the model is finetuned and you want to save and load it
# save_directory = 'saved'
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)
# tokenizer = AutoTokenizer.from_pretrained(save_directory)
# model = AutoModelForSequenceClassification.from_pretrained(save_directory)

model_name = 'oliverguhr/german-sentiment-bert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

X_train_german = ['Mit keinem guten Ergebnis', 'Das was unfair', 'Das ist gar nicht mal so gut', 'nicht so schlecht wie erwartet', 'Das war gut!', 'Sie wie ein wie Auto.']
batch = tokenizer(X_train_german, padding = True, truncation = True, max_length = 512, return_tensors = 'pt')
print('Batch german:', batch)

# If return_tensors was never put in
# batch = tokenizer(X_train_german, padding = True, truncation = True, max_length = 512)
# batch = torch.tensor(batch['input_ids'])
# Do not need to unpack the model so remove the **

with torch.no_grad():
    outputs = model(**batch)
    print('Outputs german:', outputs)
    predictions = F.softmax(outputs.logits, dim = 1)
    print('Predictions german:', predictions)
    labels = torch.argmax(predictions, dim = 1)
    print('Labels german:', labels)
    # List comprehension
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print('Labels german:', labels)