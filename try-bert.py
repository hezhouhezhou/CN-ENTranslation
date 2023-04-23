import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer

# Load the pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased')
model_zh = BertModel.from_pretrained('bert-base-chinese')


# Initialize embeddings with BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "example text to encode"
tokens = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
tokens_tensor = torch.tensor([indexed_tokens])
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor)

pooled_output = encoded_layers[-1].mean(dim=1)
#output = torch.nn.Linear(pooled_output.size(-1), num_classes)(pooled_output)
print(tokens)
# Use the BERT embeddings in a downstream model
# pooled_output = encoded_layers[-1].mean(dim=1)
# output = torch.nn.Linear(pooled_output.size(-1), num_classes)(pooled_output)