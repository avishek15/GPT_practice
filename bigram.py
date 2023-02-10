import torch
import torch.nn as nn
from torch.nn import functional as f
import torch_directml
import pandas as pd

# hyperparameters
batch_size = 32
block_size = 16
max_iters = 10000
eval_interval = 100
learning_rate = 1e-3
device = torch_directml.device()
eval_iters = 200

torch.manual_seed(1337)

# data reading
train_df = pd.read_parquet("dataset/qa_I_and_S.parquet")
test_df = pd.read_parquet("dataset/qa_VG.parquet")
chars = set()
questions = train_df.question.values
answers = train_df.answer.values

# unique characters in the training set
for q, a in zip(questions, answers):
    chars.update(set(q))
    chars.update(set(a))
chars = sorted(list(chars))
vocab_size = len(chars)

# remove characters not in training set
test_text = ' '.join(test_df.question.values)
unreg_chars = set(test_text).difference(set(chars))
rep_dict = {}
for char in unreg_chars:
    rep_dict[ord(char)] = None
test_text = test_text.translate(rep_dict)
text = ' '.join(questions)
# create a mapping for string to encoding, and encoding to string
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# train and val splits
train_data = torch.tensor(encode(text), dtype=torch.long)
val_data = torch.tensor(encode(test_text), dtype=torch.long)


# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = f.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = f.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
