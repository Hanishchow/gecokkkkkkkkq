"""
Exp 0: microgpt baseline with TRAIN/TEST split and dual loss tracking
======================================================================
Purpose: Without a TEST set, you CAN'T tell if the model is overfitting.
         Train loss going down looks great, but you need test loss to know
         if the model is actually learning or just memorizing.

Key concept: The GAP between train and test loss = overfitting.
"""
import torch, numpy as np, random, urllib.request, os
from torch import nn
from collections import Counter

device = 'cpu'
random.seed(42); torch.manual_seed(42); np.random.seed(42)

if not os.path.exists('input.txt'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt', 'input.txt')

docs = [l.strip() for l in open('input.txt').read().split('\n') if l.strip()]
random.shuffle(docs)
print(f"Total docs: {len(docs)}")

# 90/10 split
SPLIT = int(0.9 * len(docs))
docs_train = docs[:SPLIT]
docs_test = docs[SPLIT:]
print(f"Train: {len(docs_train)}, Test: {len(docs_test)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
VS = len(uchars) + 1
print(f"Vocab: {VS}")

def tokenize(s): return torch.tensor([BOS] + [uchars.index(c) for c in s] + [BOS], dtype=torch.long)

# =========================================================================
# Model: 1-layer transformer (microgpt architecture)
# =========================================================================
class GPT(nn.Module):
    def __init__(self, nl=1, ne=16, bs=16, nh=4):
        super().__init__()
        self.nl=nl; self.ne=ne; self.bs=bs; self.nh=nh; self.hd=ne//nh
        self.wte = nn.Embedding(VS, ne)
        self.wpe = nn.Embedding(bs, ne)
        self.lm  = nn.Linear(ne, VS, bias=False)
        self.Wq = nn.ModuleList([nn.Linear(ne, ne, bias=False) for _ in range(nl)])
        self.Wk = nn.ModuleList([nn.Linear(ne, ne, bias=False) for _ in range(nl)])
        self.Wv = nn.ModuleList([nn.Linear(ne, ne, bias=False) for _ in range(nl)])
        self.Wo = nn.ModuleList([nn.Linear(ne, ne, bias=False) for _ in range(nl)])
        self.fc1 = nn.ModuleList([nn.Linear(ne, 4*ne, bias=False) for _ in range(nl)])
        self.fc2 = nn.ModuleList([nn.Linear(4*ne, ne, bias=False) for _ in range(nl)])
        self.lm.weight = self.wte.weight  # weight tying
    
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.wte(x) + self.wpe(pos)
        x = x * (x.shape[-1] ** -0.5)
        
        for li in range(self.nl):
            xr = x
            q = self.Wq[li](x); k = self.Wk[li](x); v = self.Wv[li](x)
            q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
            k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
            v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
            
            # causal mask
            causal = torch.tril(torch.ones(T, T, device=x.device))
            att = q @ k.transpose(-2, -1) / (self.hd ** 0.5)
            att = att.masked_fill(causal[:T, :T] == 0, float('-inf'))
            att = torch.softmax(att, dim=-1)
            x = att @ v
            x = x.transpose(1, 2).contiguous().view(B, T, self.ne)
            x = self.Wo[li](x) + xr
            
            xr = x
            x = self.fc2[li](torch.relu(self.fc1[li](x))) + xr
        
        x = self.lm(x)
        return x
    
    def generate_names(self, temp=0.5, max_len=20, n=20):
        names = []
        for _ in range(n):
            tokens = torch.tensor([[BOS]], dtype=torch.long)
            for _ in range(max_len):
                logits = self.forward(tokens)
                next_tok = logits[0, -1] / temp
                probs = torch.softmax(next_tok, dim=-1)
                tok = torch.multinomial(probs, 1).item()
                if tok == BOS: break
                tokens = torch.cat([tokens, torch.tensor([[tok]])], dim=1)
                if tokens.shape[1] > max_len: break
            name = ''.join(uchars[t] for t in tokens[0, 1:-1] if t < len(uchars))
            names.append(name if name else '(empty)')
        return names

model = GPT(nl=1, ne=16, bs=16, nh=4).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Model: n_layer=1, n_embd=16, params={params}")

# Prepare data
train_tokens = [tokenize(d) for d in docs_train]
test_tokens = [tokenize(d) for d in docs_test]

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.85, 0.99))

def eval_loss(model, tokens_list, sample=300):
    """Sample eval for speed"""
    model.eval()
    subset = random.sample(tokens_list, min(sample, len(tokens_list)))
    total, cnt = 0.0, 0
    with torch.no_grad():
        for toks in subset:
            logits = model(toks[:-1].unsqueeze(0))
            loss = torch.nn.functional.cross_entropy(logits[0], toks[1:])
            total += loss.item(); cnt += 1
    model.train()
    return total / cnt

print(f"\n{'STEP':>6} | {'TRAIN':>8} | {'TEST':>8} | {'GAP':>8} | {'STATUS':>12}")
print("-" * 65)

results = []
for step in range(1500):
    doc = docs_train[step % len(docs_train)]
    tokens = tokenize(doc)
    optimizer.zero_grad()
    logits = model(tokens[:-1].unsqueeze(0))
    loss = torch.nn.functional.cross_entropy(logits[0], tokens[1:])
    loss.backward()
    optimizer.step()
    
    if step % 200 == 0:
        tl = eval_loss(model, train_tokens, 300)
        vx = eval_loss(model, test_tokens, 300)
        gap = vx - tl
        if gap > 0.15: status = "OVERFITTING!"
        elif gap > 0.05: status = "slight_overfit"
        elif abs(gap) < 0.03: status = "GOOD"
        else: status = "learning"
        print(f"{step:>6} | {tl:>8.4f} | {vx:>8.4f} | {gap:>+8.4f} | {status}")
        results.append({'step': step, 'train': tl, 'test': vx, 'gap': gap})

print(f"\nFinal: train={results[-1]['train']:.4f}, test={results[-1]['test']:.4f}")

# Save
import csv
with open('results/exp0_losses.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=['step','train','test','gap'])
    w.writeheader(); w.writerows(results)
print("Saved results/exp0_losses.csv")

# Generate some names
model.eval()
print("\nSample names (temp=0.5):")
for n in model.generate_names(temp=0.5, n=10):
    print(f"  {n}")
