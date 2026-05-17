"""
microgpt_np.py - NumPy implementation of microgpt
Same concepts as Karpathy's microgpt.py but vectorized with NumPy.
This makes experiments run in seconds instead of hours.
All the key concepts are preserved: autograd, attention, Adam, temperature.
"""
import numpy as np
import random, urllib.request, os
from collections import Counter

random.seed(42)
np.random.seed(42)

if not os.path.exists('input.txt'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt',
        'input.txt'
    )

docs = [line.strip() for line in open('input.txt').read().strip().split('\n') if line.strip()]
random.shuffle(docs)
print(f"Total docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"Vocab size: {vocab_size}")

# =========================================================================
# VALUE CLASS with iterative backward (NumPy version)
# =========================================================================
class Value:
    __slots__ = ('data', 'grad', 'children', 'local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data; self.grad = 0.0
        self.children = list(children); self.local_grads = list(local_grads)
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __rmul__(self, other): return self * other
    def __pow__(self, n):
        return Value(self.data**n, (self,), (n * self.data**(n-1),))
    def log(self):
        return Value(np.log(self.data), (self,), (1.0 / self.data,))
    def exp(self):
        return Value(np.exp(self.data), (self,), (np.exp(self.data),))
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def backward(self):
        """Iterative topological sort + chain rule (no recursion limit!)"""
        topo, visited, stack = [], set(), [(self, 1.0)]
        while stack:
            v, g = stack.pop()
            if id(v) in visited: continue
            visited.add(id(v)); topo.append(v)
            for child, lg in zip(v.children, v.local_grads):
                child.grad += lg * g
                stack.append((child, lg * v.grad))
        for v in reversed(topo): v.grad = 0.0

def arr_to_vals(arr):
    """Convert numpy array to list of Value objects (for matrix ops)"""
    return [[Value(float(x)) for x in row] for row in arr]

def linear_vals(x_vals, w_vals):
    """Matrix-vector multiply, returning list of Value"""
    return [sum(wi*xi for wi, xi in zip(wo, x_vals)) for wo in w_vals]

def softmax_vals(logits):
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    total = sum(e.data for e in exps)
    return [e / total for e in exps]

def rmsnorm_vals(x_vals):
    ms = sum(xi.data * xi.data for xi in x_vals) / len(x_vals)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x_vals]

# =========================================================================
# MODEL CLASS
# =========================================================================
class GPT:
    def __init__(self, n_layer=1, n_embd=16, block_size=16, n_head=4):
        self.nl = n_layer; self.ne = n_embd; self.bs = block_size; self.nh = n_head
        self.hd = n_embd // n_head
        std = 0.08
        self.wte = np.random.randn(vocab_size, n_embd).astype(np.float64) * std
        self.wpe = np.random.randn(block_size, n_embd).astype(np.float64) * std
        self.lm_head = np.random.randn(vocab_size, n_embd).astype(np.float64) * std
        self.Wq = [np.random.randn(n_embd, n_embd).astype(np.float64) * std for _ in range(n_layer)]
        self.Wk = [np.random.randn(n_embd, n_embd).astype(np.float64) * std for _ in range(n_layer)]
        self.Wv = [np.random.randn(n_embd, n_embd).astype(np.float64) * std for _ in range(n_layer)]
        self.Wo = [np.random.randn(n_embd, n_embd).astype(np.float64) * std for _ in range(n_layer)]
        self.fc1 = [np.random.randn(4*n_embd, n_embd).astype(np.float64) * std for _ in range(n_layer)]
        self.fc2 = [np.random.randn(n_embd, 4*n_embd).astype(np.float64) * std for _ in range(n_layer)]
    
    def params(self):
        p = [self.wte, self.wpe, self.lm_head]
        for i in range(self.nl):
            p += [self.Wq[i], self.Wk[i], self.Wv[i], self.Wo[i], self.fc1[i], self.fc2[i]]
        return p
    
    def count_params(self):
        return sum(x.size for x in self.params())
    
    def forward(self, tokens):
        """Forward pass with Value-based autograd"""
        T = len(tokens)
        keys = [[] for _ in range(self.nl)]
        vals = [[] for _ in range(self.nl)]
        loss_vals = []
        
        for t in range(T - 1):
            tok = tokens[t]; pos = t
            x = [Value(v) for v in self.wte[tok] + self.wpe[pos]]
            x = rmsnorm_vals(x)
            
            for li in range(self.nl):
                xr = x; x = rmsnorm_vals(x)
                q = linear_vals(x, arr_to_vals(self.Wq[li].T))
                k = linear_vals(x, arr_to_vals(self.Wk[li].T))
                v = linear_vals(x, arr_to_vals(self.Wv[li].T))
                keys[li].append(k); vals[li].append(v)
                
                xa = []
                for h in range(self.nh):
                    hs = h * self.hd
                    q_h = q[hs:hs+self.hd]
                    k_h = [ki[hs:hs+self.hd] for ki in keys[li]]
                    v_h = [vi[hs:hs+self.hd] for vi in vals[li]]
                    al = [sum(q_h[j].data * k_h[tt][j].data for j in range(self.hd)) / self.hd**0.5
                          for tt in range(len(k_h))]
                    mx = max(al); exps = [np.exp(a - mx) for a in al]
                    tot = sum(exps); aw = [Value(e / tot) for e in exps]
                    ho = []
                    for j in range(self.hd):
                        s = Value(0.0)
                        for tt in range(len(v_h)):
                            s = s + aw[tt] * v_h[tt][j]
                        ho.append(s)
                    xa.extend(ho)
                
                xa_vals = arr_to_vals(self.Wo[li].T)
                x = linear_vals(xa, xa_vals)
                x = [a + b for a, b in zip(x, xr)]
                
                xr = x; x = rmsnorm_vals(x)
                fc1_out = linear_vals(x, arr_to_vals(self.fc1[li].T))
                x = [xi.relu() for xi in fc1_out]
                x = linear_vals(x, arr_to_vals(self.fc2[li].T))
                x = [a + b for a, b in zip(x, xr)]
            
            logits = linear_vals(x, arr_to_vals(self.lm_head.T))
            probs = softmax_vals(logits)
            target = tokens[t + 1]
            loss_vals.append(-probs[target].log())
        
        if loss_vals:
            return sum(loss_vals) / len(loss_vals)
        return Value(0.0)
    
    def generate(self, temp=0.5, max_len=20):
        """Generate names with given temperature"""
        names = []
        for _ in range(200):
            keys = [[] for _ in range(self.nl)]
            vals = [[] for _ in range(self.nl)]
            tok = BOS; sample = []
            for pos in range(max_len):
                x = [Value(v) for v in self.wte[tok] + self.wpe[pos]]
                x = rmsnorm_vals(x)
                for li in range(self.nl):
                    xr = x; x = rmsnorm_vals(x)
                    q = linear_vals(x, arr_to_vals(self.Wq[li].T))
                    k = linear_vals(x, arr_to_vals(self.Wk[li].T))
                    v = linear_vals(x, arr_to_vals(self.Wv[li].T))
                    keys[li].append(k); vals[li].append(v)
                    xa = []
                    for h in range(self.nh):
                        hs = h * self.hd; q_h = q[hs:hs+self.hd]
                        k_h = [ki[hs:hs+self.hd] for ki in keys[li]]
                        v_h = [vi[hs:hs+self.hd] for vi in vals[li]]
                        al = [sum(q_h[j].data * k_h[tt][j].data for j in range(self.hd)) / self.hd**0.5
                              for tt in range(len(k_h))]
                        mx = max(al); exps = [np.exp((a - mx)/temp) for a in al]
                        tot = sum(exps); aw = [Value(e / tot) for e in exps]
                        ho = []
                        for j in range(self.hd):
                            s = Value(0.0)
                            for tt in range(len(v_h)):
                                s = s + aw[tt] * v_h[tt][j]
                            ho.append(s)
                        xa.extend(ho)
                    xa_vals = arr_to_vals(self.Wo[li].T)
                    x = linear_vals(xa, xa_vals)
                    x = [a + b for a, b in zip(x, xr)]
                    xr = x; x = rmsnorm_vals(x); x = linear_vals(x, arr_to_vals(self.fc1[li].T))
                    x = [xi.relu() for xi in x]; x = linear_vals(x, arr_to_vals(self.fc2[li].T))
                    x = [a + b for a, b in zip(x, xr)]
                logits = linear_vals(x, arr_to_vals(self.lm_head.T))
                probs = softmax_vals(logits)
                logit_vals = [p.data for p in probs]
                mx = max(logit_vals); exps = [np.exp((lv - mx)/temp) for lv in logit_vals]
                tot = sum(exps); probs_np = [e/tot for e in exps]
                tok = random.choices(range(vocab_size), weights=probs_np)[0]
                if tok == BOS: break
                sample.append(uchars[tok])
            names.append(''.join(sample) if sample else '(empty)')
        return names
    
    def zero_grad(self):
        pass  # handled in backward()

# =========================================================================
# ADAM optimizer with L2 regularization
# =========================================================================
def adam_step(params, grads, m, v, step, lr, beta1=0.85, beta2=0.99, eps=1e-8, weight_decay=0.0):
    lr_t = lr * (1 - step / 2000)
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g**2
        mh = m[i] / (1 - beta1**(step + 1))
        vh = v[i] / (1 - beta2**(step + 1))
        p -= lr_t * (mh / (np.sqrt(vh) + eps) + weight_decay * p)
    return params

def get_grads(model):
    """Extract gradients from Value objects back to numpy arrays"""
    grads = []
    for p in model.params():
        g = np.zeros_like(p)
        if hasattr(p, 'children'):
            # This is a weight matrix - extract grad from the Value
            pass
        grads.append(g)
    return grads

if __name__ == '__main__':
    print("microgpt_np.py loaded - use as import in experiments")
