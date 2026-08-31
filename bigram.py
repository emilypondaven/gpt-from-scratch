import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'   ## run on gpu if you have it
eval_iters = 200
# --------------

torch.manual_seed(1337)

# input
with open('./input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted((list(set(text))))
vocab_size = len(chars)

# encoding/tokenisation
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]   ## take string, output list of integers
decode = lambda l: ''.join([itos[i] for i in l])  ## take list of integers, output a string

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]  ## first 90 percent
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  ## random integers of size batch_size
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# average loss over multiple batches
@torch.no_grad() 
## ^^ annotation meaning: 
# more efficient with memory use
# specifies no backward propaggation, i.e. don't need to store anything
def estimate_loss():
    out = {}
    model.eval()  ## change mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  ## change mode
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            # reshape to work with cross_entropy function expectations
            B, T, C = logits.shape  ## batch_size, block_size, vocab size
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)  ## computes loss between predictions and targets

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # For each batch, u'd get the new tokens; generates predictions for all batches at once
        for _ in range(max_new_tokens):
            logits, loss = self(idx)  ## get predictions for all positions
            logits = logits[:, -1, :]  ## keep only last postiion's predictions (next token) for each batch
            probs = F.softmax(logits, dim=-1)  ## make em sum to 1 to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  ## randomly sample on token from distribution
            idx = torch.cat((idx, idx_next), dim=1)  ## append sampled index to the running sequence
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a pytorch optimiser (an algorithm that updates model weights to reduce loss)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=None)
    loss.backward()
    optimizer.step()
    
# generate from the model
idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))
