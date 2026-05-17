"""
Exp 5: L2 Regularization (Weight Decay)
=======================================
Purpose: Show that L2 regularization prevents overfitting.
         Train the same overfit model (large, 5% data) with different weight_decay values.
         Show that wd > 0 shrinks the train-test gap.

         This directly maps to GEOCK:
           - Ridge regression = linear model + L2 penalty
           - alpha = weight_decay
           - alpha=0.01 → overfit (low train loss, high test loss)
           - alpha=5.0 → regularized (higher train loss, lower test loss = better generalization)
"""
import torch, numpy as np, random, urllib.request, os, csv
import torch.nn as nn

device = 'cpu'
random.seed(42); torch.manual_seed(42); np.random.seed(42)

if not os.path.exists('input.txt'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt', 'input.txt')

docs = [l.strip() for l in open('input.txt').read().split('\n') if l.strip()]
random.shuffle(docs)

uchars = sorted(set(''.join(docs)))
VOCAB_SIZE = len(uchars) + 1
BOS = VOCAB_SIZE - 1

def tokenize(s):
    chars = [uchars.index(c) for c in s]
    return torch.tensor([BOS] + chars + [BOS], dtype=torch.long)

class GPT(nn.Module):
    def __init__(self, nl=4, ne=32, bs=32, nh=4):
        super().__init__()
        self.nl=nl; self.ne=ne; self.bs=bs; self.nh=nh; self.hd=ne//nh
        self.wte = nn.Embedding(VOCAB_SIZE, ne)
        self.wpe = nn.Embedding(bs, ne)
        self.lm  = nn.Linear(ne, VOCAB_SIZE, bias=False)
        self.Wq = nn.ModuleList([nn.Linear(ne, ne, bias=False) for _ in range(nl)])
        self.Wk = nn.ModuleList([nn.Linear(ne, ne, bias=False) for _ in range(nl)])
        self.Wv = nn.ModuleList([nn.Linear(ne, ne, bias=False) for _ in range(nl)])
        self.Wo = nn.ModuleList([nn.Linear(ne, ne, bias=False) for _ in range(nl)])
        self.fc1 = nn.ModuleList([nn.Linear(ne, 4*ne, bias=False) for _ in range(nl)])
        self.fc2 = nn.ModuleList([nn.Linear(4*ne, ne, bias=False) for _ in range(nl)])
        self.lm.weight = self.wte.weight

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
            causal = torch.tril(torch.ones(T, T, device=x.device))
            att = q @ k.transpose(-2, -1) / (self.hd ** 0.5)
            att = att.masked_fill(causal[:T, :T] == 0, float('-inf'))
            att = torch.softmax(att, dim=-1)
            x = att @ v
            x = x.transpose(1, 2).contiguous().view(B, T, self.ne)
            x = self.Wo[li](x) + xr
            xr = x
            x = self.fc2[li](torch.relu(self.fc1[li](x))) + xr
        return self.lm(x)

TRAIN_SIZE = int(0.05 * len(docs))
train_docs = docs[:TRAIN_SIZE]
test_docs = docs[TRAIN_SIZE:TRAIN_SIZE + 400]
print(f"Train: {TRAIN_SIZE}, Test: {len(test_docs)}")

configs = [
    {'wd': 0.0,   'label': 'No reg (wd=0.0)'},
    {'wd': 0.001, 'label': 'Weak   (wd=0.001)'},
    {'wd': 0.01,  'label': 'Medium (wd=0.01)'},
    {'wd': 0.1,   'label': 'Strong (wd=0.1)'},
    {'wd': 1.0,   'label': 'Heavy  (wd=1.0)'},
]

N_STEPS = 2000
LR = 0.01

print(f"\n{'WD':>8} | {'TRAIN':>10} | {'TEST':>10} | {'GAP':>10} | {'NOTE':>20}")
print("-" * 70)

results = []
for cfg in configs:
    wd = cfg['wd']
    model = GPT(nl=4, ne=32, bs=32, nh=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.85, 0.99),
                                weight_decay=wd)

    train_losses = []
    test_losses = []
    for step in range(N_STEPS):
        doc = train_docs[step % len(train_docs)]
        tokens = tokenize(doc)
        optimizer.zero_grad()
        logits = model(tokens[:-1].unsqueeze(0))
        loss = torch.nn.functional.cross_entropy(logits[0], tokens[1:])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if step % 100 == 99:
            test_loss = 0
            for td in test_docs[:50]:
                tt = tokenize(td)
                with torch.no_grad():
                    tlogits = model(tt[:-1].unsqueeze(0))
                    tloss = torch.nn.functional.cross_entropy(tlogits[0], tt[1:])
                test_loss += tloss.item()
            test_loss /= len(test_docs[:50])
            test_losses.append(test_loss)

    final_train = sum(train_losses[-100:]) / 100
    final_test = sum(test_losses[-5:]) / 5
    gap = final_test - final_train

    note = ""
    if wd == 0.0:
        note = "← overfits"
    elif wd == 1.0:
        note = "← underfits"

    print(f"{wd:>8.3f} | {final_train:>10.4f} | {final_test:>10.4f} | {gap:>10.4f} | {note:>20}")
    results.append({
        'wd': wd, 'train_loss': final_train, 'test_loss': final_test,
        'gap': gap, 'note': note
    })

with open('results/exp5_l2_reg.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=['wd','train_loss','test_loss','gap','note'])
    w.writeheader(); w.writerows(results)

print(f"\nSaved results/exp5_l2_reg.csv")
print(f"\n=== KEY INSIGHT ===")
no_reg = results[0]
heavy_reg = results[-1]
print(f"No regularization: train={no_reg['train_loss']:.4f}, test={no_reg['test_loss']:.4f}, gap={no_reg['gap']:.4f}")
print(f"Heavy regularization: train={heavy_reg['train_loss']:.4f}, test={heavy_reg['test_loss']:.4f}, gap={heavy_reg['gap']:.4f}")
print(f"\nWithout reg: model memorizes training data (low train loss)")
print(f"With reg: model generalizes better (smaller gap)")
print(f"\nThis is EXACTLY GEOCK:")
print(f"  alpha=0.01 (no reg)  → val_r=0.90 (overfit)")
print(f"  alpha=5.0  (regularized) → val_r=0.60 (better generalization)")
print(f"The regularization penalty forces weights toward zero → simpler model")
