import random
import requests
import time
import re

# Perplexity API call setup
def call_perplexity_api(prompt, api_key):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar-pro",
        "messages": [{"role":"system","content":"You are a helpful assistant."},
                     {"role":"user","content":prompt}],
        "max_tokens":40000,
        "temperature":0.5
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def parse_feedback_for_improvements(feedback):
    adjustments = []
    if "dropout" in feedback.lower():
        adjustments.append("adjust_dropout")
    if "learning rate" in feedback.lower() or "lr" in feedback.lower():
        adjustments.append("adjust_lr")
    if "layer" in feedback.lower() or "architecture" in feedback.lower():
        adjustments.append("adjust_layer")
    return adjustments if adjustments else ["general_tweak"]

def apply_adjustments(code_lines, adjustments):
    lines = code_lines.copy()
    for adj in adjustments:
        if adj == "adjust_dropout":
            for i, line in enumerate(lines):
                if '.Dropout(' in line or 'nn.Dropout(' in line:
                    new_p = round(random.uniform(0.1, 0.4), 2)
                    parts = line.split('(')
                    if len(parts) > 1:
                        lines[i] = parts[0] + f'({new_p})  # adjusted dropout'
                    break
        elif adj == "adjust_lr":
            # Example: find and scale learning rate
            for i, line in enumerate(lines):
                if 'learning_rate' in line or 'lr =' in line:
                    try:
                        old_lr = float(re.findall(r"\d*\.\d+|\d+", line)[0])
                        new_lr = old_lr * random.uniform(0.7, 1.3)
                        lines[i] = f"learning_rate = {new_lr:.6f}  # adjusted learning rate"
                        break
                    except:
                        continue
        elif adj == "adjust_layer":
            for i, line in enumerate(lines):
                if 'nn.Linear' in line:
                    nums = re.findall(r'\d+', line)
                    if len(nums) >= 2:
                        in_sz = int(nums[0])
                        out_sz = int(nums[1])
                        new_out = int(out_sz * random.uniform(0.8, 1.2))
                        lines[i] = f"nn.Linear({in_sz}, {new_out})  # adjusted layer size"
                        break
        elif adj == "general_tweak":
            # Minor random tweak: delete random line or add comment
            if random.random() < 0.5:
                non_empty = [i for i,l in enumerate(lines) if l.strip()]
                if non_empty:
                    idx = random.choice(non_empty)
                    lines.pop(idx)
            else:
                idx = random.randint(0, len(lines)-1)
                lines[idx] += "  # tweak"
    return lines

def iterative_improve(original_code, api_key, filename_prefix="Model", iterations=3):
    current_code = original_code.split('\n')
    results = []

    for it in range(iterations):
        # Step 1: generate randomized variant or improved code
        if it == 0:
            variant_lines = current_code
        else:
            variant_lines = apply_adjustments(current_code, improvement_adjs)

        # Step 2: convert to string and save
        variant_code_str = "\n".join(variant_lines)
        filename = f"{filename_prefix}_iter_{it+1}.py"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(variant_code_str)

        # Step 3: ask Perplexity AI for feedback
        prompt = (f"Evaluate this Python AI neural network code and make improvements, write the whole code:\n\n{variant_code_str}\n")
        try:
            print(f"Iteration {it+1}: Sending code to Perplexity AI...")
            feedback = call_perplexity_api(prompt, api_key)
            print(f"Feedback received:\n{feedback}\n")
        except Exception as e:
            feedback = f"API error: {e}"
            print(feedback)

        # Step 4: parse feedback to decide improvements
        improvement_adjs = parse_feedback_for_improvements(feedback)

        current_code = variant_lines
        results.append((filename, feedback))

        time.sleep(1)  # polite pause to avoid rate limits

    return results

# Example usage (fill in your API key and your full neural net code as a string):
if __name__ == "__main__":
    API_KEY = ""
    your_code = '''

# ----------------------------  FGNN  MEMORY-OPT  ----------------------------
import torch, torch.nn as nn
import numpy as np, random, math
from collections import defaultdict, Counter

# --------------------------------------------------------------------------- #
#                         Helper: precision & device                          #
# --------------------------------------------------------------------------- #
DTYPE_ACT  = torch.float16         # everything heavy = FP16
DTYPE_SMALL= torch.float32         # small / critical ops
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
#                 Functionally-Graded parameter generator (FGP)               #
# --------------------------------------------------------------------------- #
class FGParams:
    def __init__(self, length, tau_base=0.8, smooth=2.0):
        self.tau_base = tau_base
        self.smooth   = smooth
        # store on CPU; cast to needed dtype on first use
        x = np.linspace(-3, 3, length, dtype=np.float32)
        t = np.linspace(0, 1, length, dtype=np.float32)
        self.spatial  = torch.sigmoid(torch.from_numpy(x))
        self.temporal = torch.from_numpy(
            np.exp(-t) * (1 + 0.3*np.sin(4*np.pi*t))
        )

    # quick access as float (for NumPy maths) or tensor (for Torch maths)
    def s(self, i): return self.spatial[min(i, len(self.spatial)-1)].item()
    def t(self, i): return self.temporal[min(i, len(self.temporal)-1)].item()

# --------------------------------------------------------------------------- #
#                            Corpus / n-gram storage                          #
# --------------------------------------------------------------------------- #
class TextProcessor:
    def __init__(self):
        self.word_counts   = Counter()
        self.bigram_counts = defaultdict(Counter)    # w1  -> Counter(w2)
        self.trigram       = defaultdict(Counter)    # (w1,w2)->Counter(w3)
        self.trans_probs   = defaultdict(list)
        self.geom = {}

    def ingest(self, text:str):
        words = text.lower().split()
        self.word_counts.update(words)
        for i in range(len(words)-2):
            w1,w2,w3 = words[i:i+3]
            self.bigram_counts[w1][w2] += 1
            self.trigram[(w1,w2)][w3] += 1
        # unigram transition probs
        for w1,ctr in self.bigram_counts.items():
            tot = sum(ctr.values())
            self.trans_probs[w1]=[(w2,c/tot) for w2,c in ctr.items()]

    def uni(self, w):           return self.trans_probs.get(w, [])
    def tri(self, w1,w2):       return self.trigram.get((w1,w2),{})

# --------------------------------------------------------------------------- #
#                               Neural network                                #
# --------------------------------------------------------------------------- #
class FGNNTextGen(nn.Module):
    def __init__(self, in_size=100, hid=128, vocab=1000, max_tr=40):
        super().__init__()
        self.enc   = nn.Linear(in_size, hid)
        self.hid   = nn.Linear(hid, hid)
        self.out   = nn.Linear(hid, vocab)
        self.drop  = nn.Dropout(0.2)
        self.max_tr= max_tr
        self.v_thresh = nn.Parameter(torch.tensor(1.0, dtype=DTYPE_SMALL))
        self.tp = TextProcessor()
        self.fb = ['the','and','is','in','to','of','a','for','on','with']
        self._init_w()

    def _init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, spikes):
        x = torch.tanh(self.enc(spikes))
        x = self.drop(x)
        x = torch.tanh(self.hid(x))
        x = self.drop(x)
        return torch.softmax(self.out(x), dim=-1)

    # --- helpers ----------------------------------------------------------- #
    def _multi_trans(self, words):
        if len(words)<2: return self.tp.uni(words[-1])
        ctx=(words[-2],words[-1]); tri=self.tp.tri(*ctx)
        if tri: return [(w,c/sum(tri.values())) for w,c in tri.items()]
        return self.tp.uni(words[-1])

    # mask for dataset-seen words
    def _mask(self, cand, ctx_w1):
        seen=set(self.tp.bigram_counts[ctx_w1])|{ctx_w1}
        return np.array([w in seen for w in cand])

    # ----------------------------------------------------------------------- #
    #                        Main generation (memory-lite)                    #
    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, spikes, seed="", length=50, hidden=None, tau=0.8):
        n_steps = length
        # 1) cast heavy tensors to FP16 & DEVICE, stream row-by-row
        spikes = spikes.to(device=DEVICE, dtype=DTYPE_ACT)
        if hidden is not None:
            hidden = hidden.to(device=DEVICE, dtype=DTYPE_ACT)

        fg = FGParams(n_steps, tau)
        sel_probs = self.forward(spikes)          # (B,L,vocab) FP16
        v_mem = torch.zeros_like(spikes, device=DEVICE)
        i_syn = torch.zeros_like(spikes, device=DEVICE)

        # seed processing ---------------------------------------------------- #
        cur = seed.lower().split() if seed else [random.choice(self.fb)]
        cur = [w for w in cur if w]; gen = cur.copy()

        # advantage completeness
        high = (sel_probs>tau).float().mean().item()
        comp = torch.sigmoid(torch.tensor((high-0.05)*20, dtype=DTYPE_SMALL))
        #print(f"High-prob steps {100*high:.1f} %, completeness {comp:.3f}")

        for i in range(n_steps):
            # STREAM one spike row
            x = spikes[i]

            # graded LIF params
            beta  = 0.8+0.4*fg.s(i); alpha=0.1+0.3*fg.t(i)
            i_syn = alpha*i_syn + x
            v_mem = beta*v_mem + i_syn*torch.sigmoid(v_mem*fg.smooth)

            # graded threshold + spikes
            thr = 1.0+0.3*math.sin(2*math.pi*fg.s(i))
            psp = torch.sigmoid((v_mem-thr)*fg.smooth)
            gumb = -torch.log(-torch.log(torch.rand_like(psp)+1e-8)+1e-8)
            spikes_i = torch.sigmoid((torch.log(psp+1e-8)-torch.log1p(-psp)+gumb)/0.1)

            # synthetic reset (vectorised, low mem)
            sf = fg.s(i)
            diff = torch.linspace(0,1,len(x), device=DEVICE, dtype=DTYPE_ACT)
            rew = torch.abs(diff*100 - diff*20000)
            combined = torch.clamp(rew*(0.8+0.4*sf), -2, 2)
            reset_s  = torch.sigmoid(spikes_i*(3+2*sf))
            v_mem = v_mem*(1-reset_s) + combined*reset_s.unsqueeze(0)

            # transition candidates ----------------------------------------- #
            trans = self._multi_trans(cur)[:self.max_tr]
            if not trans: trans=self.tp.uni(cur[-1]) or [(random.choice(self.fb),1)]
            words, wts = zip(*trans); wts=np.array(wts, dtype=np.float32)

            base_p = sel_probs[i].mean().item()
            hid_p  = torch.sigmoid(hidden[i]).mean().item() if hidden is not None else None
            prob_use = hid_p if hid_p is not None else base_p

            # graded weight modulation
            wts = fg.smooth_mod = 1.2+0.8*fg.s(i) if prob_use>=tau else 0.6+0.3*fg.s(i)
            wts = np.array(wts)*np.ones(len(words), dtype=np.float32)

            # dataset constraint if incompleteness high
            if comp<0.7 and len(cur)>=2:
                mask=self._mask(words,cur[-2])
                wts = np.where(mask,wts,wts*0.3)

            # softmax sample
            wts /= wts.sum(); nxt=np.random.choice(words,p=wts)
            gen.append(nxt); cur.append(nxt); cur=cur[-3:]

            # free temporaries
            del spikes_i, reset_s
        return " ".join(gen)

# --------------------------------------------------------------------------- #
#                               Quick demo                                    #
# --------------------------------------------------------------------------- #
def demo():
    model = FGNNTextGen().to(DEVICE, dtype=DTYPE_ACT)
    with open(input("Filename: "), 'r', encoding='utf-8') as f:
                  
        corpus = f.read()
    model.tp.ingest(corpus)

    B,T = 520,100
    spikes = torch.randn(B, T, dtype=DTYPE_ACT)
    hidden = torch.randn(B, 128, dtype=DTYPE_ACT)
    while True:
        txt = model.generate(spikes, seed=input("USER: "), length=520, hidden=hidden)
        print("\nGenerated:\n", txt)

if __name__ == "__main__":
    demo()

'''
    results = iterative_improve(your_code, API_KEY, "SimpleNet", iterations=3)
    for fname, fb in results:
        print(f"Saved {fname} with feedback:\n{fb}\n{'-'*40}")
