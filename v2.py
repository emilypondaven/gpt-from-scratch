import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'   ## run on gpu if you have it
eval_iters = 200
n_embd = 32
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

class Head(nn.Module):
    """ one head of self attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  ## (B,T,C)
        q = self.query(x)  ## (B,T,C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out
    
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        
    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
    
# super simple bigram model
class BigramLanguageModel(nn.Module):
    ## deep learning nets suffer from optimisation issues
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),  # i.e. 4 heads of 8-dimensional self-attention
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)  ## (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  ## (T,C)
        x = tok_emb + pos_emb  ## (B, T, C)
        x = self.blocks(x)  ## (B, T, C)
        logits = self.lm_head(tok_emb)  ## (B,T,vocab_size)
        
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
            idx_cond = idx[:, -block_size:]  ## crop idx to the last block_size tokens
            logits, loss = self(idx_cond)  ## get predictions for all positions
            logits = logits[:, -1, :]  ## keep only last postiion's predictions (next token) for each batch
            probs = F.softmax(logits, dim=-1)  ## make em sum to 1 to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  ## randomly sample on token from distribution
            idx = torch.cat((idx, idx_next), dim=1)  ## append sampled index to the running sequence
        return idx

model = BigramLanguageModel()
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
