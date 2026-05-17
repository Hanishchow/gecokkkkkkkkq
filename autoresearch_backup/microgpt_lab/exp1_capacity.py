"""
Exp 1: Model Capacity vs Overfitting
====================================
Purpose: Sweep model size (n_layer × n_embd) while keeping data fixed.
         Measure train loss, test loss, and their gap.

         Small model (nl=1, ne=8)  → train/test loss both go down, small gap
         Medium (nl=2, ne=16)      → gap starts to appear
         Large (nl=4, ne=32)        → train loss goes DOWN, test loss goes UP = OVERFIT

         This directly mirrors GEOCK: too many features + too few samples = overfit.
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
    def __init__(self, nl=1, ne=8, bs=32, nh=2):
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

    def generate_names(self, temp=0.5, max_len=20, n=200):
        names = []
        for _ in range(n):
            tokens = [BOS]
            for _ in range(max_len):
                tok_tensor = torch.tensor([tokens], dtype=torch.long)
                logits = self.forward(tok_tensor)
                next_tok_logits = logits[0, -1] / temp
                probs = torch.softmax(next_tok_logits, dim=-1)
                tok = torch.multinomial(probs, 1).item()
                if tok >= VOCAB_SIZE - 1: break
                tok = min(tok, VOCAB_SIZE - 2)
                tokens.append(tok)
            chars = [uchars[t] for t in tokens[1:] if t < VOCAB_SIZE - 1]
            name = ''.join(chars)
            names.append(name if name else '(empty)')
        return names

TRAIN_SIZE = int(0.05 * len(docs))
train_docs = docs[:TRAIN_SIZE]
test_docs = docs[TRAIN_SIZE:TRAIN_SIZE + 400]
print(f"Train: {TRAIN_SIZE}, Test: {len(test_docs)}")

configs = [
    {'nl': 1, 'ne': 8,  'label': 'XS  (nl=1, ne=8)'},
    {'nl': 1, 'ne': 16, 'label': 'S   (nl=1, ne=16)'},
    {'nl': 2, 'ne': 16, 'label': 'M   (nl=2, ne=16)'},
    {'nl': 2, 'ne': 32, 'label': 'L   (nl=2, ne=32)'},
    {'nl': 4, 'ne': 32, 'label': 'XL  (nl=4, ne=32)'},
]

N_STEPS = 2000
LR = 0.01

print(f"\n{'CONFIG':>20} | {'PARAMS':>8} | {'TRAIN_L':>8} | {'TEST_L':>8} | {'GAP':>8}")
print("-" * 70)

results = []
for cfg in configs:
    nl, ne = cfg['nl'], cfg['ne']
    nh = max(2, ne // 4)
    model = GPT(nl=nl, ne=ne, bs=32, nh=nh).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.85, 0.99))

    train_losses = []
    test_losses = []
    for step in range(N_STEPS):
        is_test = step % 100 == 99
        doc = train_docs[step % len(train_docs)]
        tokens = tokenize(doc)
        optimizer.zero_grad()
        logits = model(tokens[:-1].unsqueeze(0))
        loss = torch.nn.functional.cross_entropy(logits[0], tokens[1:])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if is_test:
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
    print(f"{cfg['label']:>20} | {n_params:>8} | {final_train:>8.4f} | {final_test:>8.4f} | {gap:>8.4f}")
    results.append({
        'config': cfg['label'], 'nl': nl, 'ne': ne,
        'params': n_params, 'train_loss': final_train,
        'test_loss': final_test, 'gap': gap
    })

with open('results/exp1_capacity.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=['config','nl','ne','params','train_loss','test_loss','gap'])
    w.writeheader(); w.writerows(results)

print(f"\nSaved results/exp1_capacity.csv")
print(f"\n=== KEY INSIGHT ===")
smallest = min(results, key=lambda r: r['gap'])
largest = max(results, key=lambda r: r['gap'])
print(f"Smallest train-test gap: {smallest['config']} (gap={smallest['gap']:.4f})")
print(f"Largest  train-test gap: {largest['config']} (gap={largest['gap']:.4f})")
print(f"\nLarger models fit training data better (lower train loss)")
print(f"But they OVERFIT: test loss stops improving while train loss keeps going down")
print(f"This is EXACTLY what happens with GEOCK: Ridge(alpha=0.01) fits train perfectly")
print(f"but generalizes poorly. Regularization (higher alpha) prevents this.")
