! pip install tiktoken

import torch
from torch.utils.data import DataLoader
import tiktoken

# BPE for tokenization
tokenizer = tiktoken.get_encoding("gpt2")

def create_gpt_dataset(txt, tokenizer, max_length, stride):
    input_ids, target_ids = [], []
    # Tokenize the entire text
    token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
    # and ----> established
    # and established ----> himself
    # and established himself ----> in
    # and established himself in ----> a
    for i in range(0, len(token_ids) - max_length, stride):
        input_chunk = token_ids[i:i + max_length]
        target_chunk = token_ids[i + 1: i + max_length + 1]
        input_ids.append(torch.tensor(input_chunk))
        target_ids.append(torch.tensor(target_chunk))
    return input_ids, target_ids

def create_dataloader_v1(txt, batch_size=4, max_length=256,stride=128, shuffle=True, drop_last=True, num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids, target_ids = create_gpt_dataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        list(zip(input_ids, target_ids)), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)
    return dataloader


# Reading train file
with open("/content/train.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


vocab_size = 50257 # BPE vocab size
output_dim = 256 # Dimension for embeded vectors
context_length = 1024 # Sliding window
max_length = 4 # Number of tokens in each vector

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)



class TransformerGenerator(torch.nn.Module):
    def __init__(self, vocab_size, output_dim, context_length, num_layers=2, num_heads=2, dropout=0.1):
        super(TransformerGenerator, self).__init__()
        # Creating Embedding Layers
        self.token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        self.pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
        # 2 Transformers and 1 Fully Connected Layer
        self.transformer_encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, num_heads, dim_feedforward=2048, dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layers, num_layers)
        self.fc = torch.nn.Linear(output_dim, vocab_size)

    def forward(self, x):
        # Converting Tokens to Embedded Vectors
        token_embeddings = self.token_embedding_layer(x)
        # Positional Embedding
        pos_embeddings = self.pos_embedding_layer(torch.arange(x.size(1)))
        # Input for Model
        input_embeddings = token_embeddings + pos_embeddings
        input_embeddings = input_embeddings.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, embed_dim)
        transformer_output = self.transformer_encoder(input_embeddings)
        transformer_output = transformer_output.permute(1, 0, 2)  # Reshape to (batch_size, seq_len, embed_dim)
        logits = self.fc(transformer_output)
        return logits

# Creating The Model
generator_model = TransformerGenerator(vocab_size, output_dim, context_length)

# Loss Function and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(generator_model.parameters(), lr=0.001)

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    for batch in dataloader:
        x, y = batch
        optimizer.zero_grad()
        logits = generator_model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')



import torch

def generate_text(model, tokenizer, seed_text, max_length=100, temperature=1.0):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        token_ids = tokenizer.encode(seed_text)
        for _ in range(max_length):
            input_ids = torch.tensor(token_ids).unsqueeze(0)
            logits = model(input_ids)
            logits = logits[0, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            token_ids.append(next_token_id.item())
        generated_text = tokenizer.decode(token_ids)
    return generated_text

# Input Sample Text
seed_text = "hello batman"
generated_text = generate_text(generator_model, tokenizer, seed_text, max_length=15, temperature=0.7)
print(generated_text)


# Save the model
torch.save(generator_model.state_dict(), 'generator_model.pth')
