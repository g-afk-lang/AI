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
