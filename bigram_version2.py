import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' # run on a gpu if u have it to make it faster
eval_iters = 200
n_embd = 32
# ------------

torch.manual_seed(1337) # seed can be anything.

# 1. READ DATA
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# 2. ENCODER AND DECODER
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# 3. CREATE Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# 4. DATA LOADER 
# inp = x, batch of inputs, targets = y
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # if the device is cuda, while loading the data we need to move the data to the device so that all calculations happen inside the gpu and its faster.
    return x, y
# 5. ESTIMATE LOSS
@torch.no_grad() # we will not call .backward on these which makes it more memory efficient since it wont need to store all the intermediate variables because we are not calling it backward.
def estimate_loss():
    out = {}
    model.eval() # evaluation phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item() # earlier we used to print the loss.item of each iteration and thats more or less lucky
            # now it averages up the loss over multiple batches evaluating eval_iter times for both splits making it lot less noisy
        out[split] = losses.mean()
    model.train() # training phase
    return out


# 6. BIGRAM LANGUAGE MODEL
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # n_embd is short for number of embedding dimensions
        # this will provide us with the embedding table and only 32 dimensional embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # embedding of block size by n_embd
        self.lm_head = nn.Linear(n_embd, vocab_size) # to go from token embeddings to logits we will need the linear layer.
        # lm_head is short for language modelling head.
        # this just creates a spurious layer of interaction through a linear layer
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # till now we have taken the indices and we have encoded them based on the identity of the tokens inside idx
        # but we could just also encode the indices along with the positions.
        token_embd = self.token_embedding_table(idx) # this will give us token embeddings nd produce (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device = device)) # (T, C), basically integers from 0 to T-1
        # all of those integers from 0 to T-1 get embedded through the table to create a T x C
        x = token_embd + pos_embd # (B, T,C) = (B, T, C) + (T, C), this gets right aligned and a new dimension fo 1 gets added and it gets broadcasted acros batch so at this point, x holds not just the token identities but the positions at which these tokens occur 
        # This aint matter that much at this point since now its a simple bigram model but it will start to make sense when we work on the self attention model,
        # Now, its all translation invariant at this stage so the info wouldn't help but as we work on the self attention block we will see that this starts to matter.
        logits = self.lm_head(x) # (B,T,C=vocab_size)
        # but this C and the other C is not the same. the C in token_embd is the n_embd  and the C here is the vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device) # move the model parameters to the device


# 7. PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # 
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))  
