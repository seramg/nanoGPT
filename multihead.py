import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6 # layer of blocks we are going to have
dropout = 0.2 # for every forward backward pass 20% of all of these intermediate calculations are disabled and dropped to 0.

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # randomly preventing some nodes from communicating
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel""" 
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # projection layer that foes back in the residual pathway
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # output of the self attention
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    """a simple linear layer followed by a non linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout) # before the residual connection back right before the connection back into the residual pathway so we can drop out that as the last layer
        )
    
    def forward(self, x): # on a per token level, all of the tokens do this independently so self attention is the communication
        # once they gather day, they need to think on the data
        return self.net(x)
    
    
class Block(nn.Module):
    """Transformer block: communication followed by computation is grouped and replicates them"""
    # intersperse the communication with the computation
    
    def __init__(self, n_embd, n_head): # group size in group convolution
        # the block basically intersperses communication and then computation.
        super().__init__()
        head_size = n_embd // n_head # n_embd is 32 and head size should be 8 so divided by 4 ie, n_head
        self.sa = MultiHeadAttention(n_head, head_size) # communication is done
        self.ffwd = FeedForward(n_embd) # computation is done
        self.ln1 = nn.LayerNorm(n_embd) # 32, when normalisation is done the mean and variance are taken over 32 number so the batch and the time act as batch dimensions 
        self.ln2 = nn.LayerNorm(n_embd)
        
        def forward(self, x):
            # # done independently on all the tokens.
            # x = x + self.sa(x) # we fork off and do some communication nd we come back
            # x = x + self.ffwd(x) # optimisation, sum just distributes the gradients
            # # we fork off and do some computation nd we come back
            
            x = x + self.sa(self.ln1(x)) # per token transformation that just normalizes the features and makes them a unit mean, unit Gaussian at initialization 
            c = x + self.ffwd(self.ln2(x))
            
            return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # # 4 communication channel
        self.sa_head = MultiHeadAttention(4, n_embd//4) # ie, 4 heads of 8 dimensional self attention
        self.ffwd = FeedForward(n_embd)
        
        # sequential application of blocks
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4), # size n_embd is 32 it must be reduced to 8 for it to be computed with n_head therefore head_size = n_embd // n_head
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd) # just at the end of the transformer and before the final linear layer
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device = device))

        x = token_embd + pos_embd
        
        x = self.sa_head(x) # apply 1 head of self attention (B,T,C)
        x = self.ffwd(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -min(block_size, idx.shape[1]):]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
