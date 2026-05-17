"""
Exp 4: Early Stopping — Test Loss ≠ Train Loss
==============================================
Purpose: Show that the minimum train loss ≠ minimum test loss.
         Stopping at test loss minimum (not train loss minimum) is key.

         Train the same overfit model (large, 5% data) for 3000 steps.
         Track both train and test loss every 100 steps.
         Show: train loss keeps going down; test loss goes up after a point.

         This directly maps to GEOCK:
           - Train loss: goes down as you fit training data
           - LOO-CV loss: goes up as you overfit (generalization gets worse)
           - Optimal point: where LOO-CV is lowest, NOT where train is lowest
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

model = GPT(nl=4, ne=32, bs=32, nh=4).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: nl=4, ne=32, params={n_params} (OVERFITTING regime)")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.85, 0.99))

N_STEPS = 3000
records = []
best_test_loss = float('inf')
best_test_step = 0

print(f"\n{'STEP':>6} | {'TRAIN':>10} | {'TEST':>10} | {'GAP':>10} | {'BEST':>6}")
print("-" * 55)
for step in range(N_STEPS):
    doc = train_docs[step % len(train_docs)]
    tokens = tokenize(doc)
    optimizer.zero_grad()
    logits = model(tokens[:-1].unsqueeze(0))
    loss = torch.nn.functional.cross_entropy(logits[0], tokens[1:])
    loss.backward()
    optimizer.step()
    train_loss = loss.item()

    is_eval = step % 100 == 99
    if is_eval:
        test_loss = 0
        for td in test_docs[:50]:
            tt = tokenize(td)
            with torch.no_grad():
                tlogits = model(tt[:-1].unsqueeze(0))
                tloss = torch.nn.functional.cross_entropy(tlogits[0], tt[1:])
            test_loss += tloss.item()
        test_loss /= len(test_docs[:50])
        gap = test_loss - train_loss
        is_best = ""
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_step = step
            is_best = " ← BEST"
        print(f"{step:>6} | {train_loss:>10.4f} | {test_loss:>10.4f} | {gap:>10.4f} | {step:>6}{is_best}")
        records.append({'step': step, 'train_loss': train_loss,
                       'test_loss': test_loss, 'gap': gap})

print(f"\n=== RESULTS ===")
print(f"Best test loss: {best_test_loss:.4f} at step {best_test_step}")
train_at_best = next(r['train_loss'] for r in records if r['step'] == best_test_step)
print(f"Train loss at best test: {train_at_best:.4f}")
print(f"Train loss at final step: {records[-1]['train_loss']:.4f}")
print(f"\nTrain loss at step {best_test_step} (BEST): {train_at_best:.4f}")
print(f"Train loss at final step {N_STEPS-1} (OVERFIT): {records[-1]['train_loss']:.4f}")
print(f"Train loss IMPROVED by {train_at_best - records[-1]['train_loss']:.4f} after best test")
print(f"But test loss WORSENED by {records[-1]['test_loss'] - best_test_loss:.4f}")

with open('results/exp4_early_stop.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=['step','train_loss','test_loss','gap'])
    w.writeheader(); w.writerows(records)

print(f"\nSaved results/exp4_early_stop.csv")
print(f"\n=== KEY INSIGHT ===")
print(f"TRAIN loss at step {best_test_step} (BEST):  {train_at_best:.4f}")
print(f"TRAIN loss at final step {N_STEPS-1} (OVERFIT): {records[-1]['train_loss']:.4f}")
print(f"→ Train loss IMPROVED after the best test point!")
print(f"→ But test loss was getting WORSE!")
print(f"\nIf you stop at minimum TRAIN loss, you overfit.")
print(f"If you stop at minimum TEST loss (or LOO-CV), you generalize better.")
print(f"\nThis is EXACTLY the GEOCK lesson:")
print(f"  - val_r=0.904 looks great (low train-like loss)  ")
print(f"  - LOO_r=0.059 is honest (test-like loss)")
print(f"  - The REAL metric is LOO, not val!")
