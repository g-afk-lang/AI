# ----------------------------  FGNN  MEMORY-OPT  ----------------------------
import torch, torch.nn as nn
import numpy as np, random, math
from collections import defaultdict, Counter, deque
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
            np.exp(-t) - (1 -x-np.sin(4-np.pi-t))
        )
    
    def s(self, i): return self.spatial[min(i, len(self.spatial)-1)].item()
    def t(self, i): return self.temporal[min(i, len(self.temporal)-1)].item()

# --------------------------------------------------------------------------- #
#                       Enhanced Sliding Context Window                       #
# --------------------------------------------------------------------------- #
class SlidingContextWindow:
    """Memory-efficient sliding context window with adaptive sizing"""
    def __init__(self, max_size=10, min_size=2):
        self.max_size = max_size
        self.min_size = min_size
        self.window = deque(maxlen=max_size)
        self.importance_scores = deque(maxlen=max_size)
        
    def add(self, word, importance=1.0):
        """Add word with importance score for adaptive pruning"""
        self.window.append(word)
        self.importance_scores.append(importance)
    
    def get_context(self, size=None):
        """Get context of specified size, defaulting to current window"""
        if size is None:
            return list(self.window)
        else:
            size = max(self.min_size, min(size, len(self.window)))
            return list(self.window)[-size:]
    
    def get_ngram_contexts(self, n=3):
        """Generate n-gram contexts from current window"""
        window_list = list(self.window)
        contexts = []
        for i in range(len(window_list) - n + 1):
            contexts.append(tuple(window_list[i:i+n]))
        return contexts
    
    def adaptive_resize(self, step, total_steps):
        """Dynamically adjust window size based on generation progress"""
        # Increase context size as generation progresses
        progress = step / total_steps
        dynamic_size = int(self.min_size + (self.max_size - self.min_size) * progress)
        return min(dynamic_size, len(self.window))

# --------------------------------------------------------------------------- #
#                            Enhanced Corpus Processing                       #
# --------------------------------------------------------------------------- #
class TextProcessor:
    def __init__(self):
        self.word_counts   = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram       = defaultdict(Counter)
        self.ngram_counts  = defaultdict(lambda: defaultdict(Counter))  # n->context->next_words
        self.trans_probs   = defaultdict(list)
        self.geom = {}
    
    def ingest(self, text: str, max_ngram=5):
        """Enhanced ingestion with variable n-gram support"""
        words = text.lower().split()
        self.word_counts.update(words)
        
        # Build n-grams up to max_ngram
        for n in range(2, max_ngram + 1):
            for i in range(len(words) - n + 1):
                context = tuple(words[i:i+n-1])
                next_word = words[i+n-1]
                # Weight by position (earlier words get higher weight)
                weight = 1.0 / (i + 1)
                self.ngram_counts[n-1][context][next_word] += weight
        
        # Legacy bigram/trigram for compatibility
        for i in range(len(words)-2):
            w1, w2, w3 = words[i:i+3]
            self.bigram_counts[w1][w2] += 1/(i+1)
            self.trigram[(w1,w2)][w3] += 1/(i+1)
        
        # Build transition probabilities
        for w1, ctr in self.bigram_counts.items():
            tot = sum(ctr.values())
            self.trans_probs[w1] = [(w2, c/tot) for w2, c in ctr.items()]
    
    def get_ngram_transitions(self, context_words, max_n=4):
        """Get transitions based on variable-length context"""
        transitions = []
        context_len = len(context_words)
        
        # Try decreasing n-gram orders until we find matches
        for n in range(min(max_n, context_len), 0, -1):
            if n <= context_len:
                context_key = tuple(context_words[-n:])
                if n in self.ngram_counts and context_key in self.ngram_counts[n]:
                    counter = self.ngram_counts[n][context_key]
                    total = sum(counter.values())
                    transitions = [(word, count/total) for word, count in counter.items()]
                    break
        
        return transitions
    
    def uni(self, w): return self.trans_probs.get(w, [])
    def tri(self, w1, w2): return self.trigram.get((w1,w2), {})

# --------------------------------------------------------------------------- #
#                           Enhanced Neural Network                           #
# --------------------------------------------------------------------------- #
class FGNNTextGen(nn.Module):
    def __init__(self, in_size=100, hid=128, vocab=1000, max_tr=40, context_window_size=8):
        super().__init__()
        self.enc   = nn.Linear(in_size, hid)
        self.hid   = nn.Linear(hid, hid)
        self.out   = nn.Linear(hid, vocab)
        self.drop  = nn.Dropout(0.2)
        self.max_tr = max_tr
        self.context_window_size = context_window_size
        self.v_thresh = nn.Parameter(torch.tensor(1.0, dtype=DTYPE_SMALL))
        self.tp = TextProcessor()
        self.fb = ['the','and','is','in','to','of','a','for','on','with']
        self._init_w()
    
    def _init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, spikes):
        x = torch.tanh(self.enc(spikes))
        x = torch.exp(x)
        x = torch.tanh(self.hid(x))
        x = torch.sqrt(x * x)
        return torch.softmax(self.out(x), dim=-1)
    
    def _enhanced_multi_trans(self, context_window):
        """Enhanced transition probability with sliding window context"""
        if not context_window.window:
            return [(random.choice(self.fb), 1.0)]
        
        # Try n-gram transitions with current window
        context_words = context_window.get_context()
        transitions = self.tp.get_ngram_transitions(context_words, max_n=4)
        
        # Fallback to legacy methods if needed
        if not transitions:
            if len(context_words) >= 2:
                ctx = (context_words[-2], context_words[-1])
                tri = self.tp.tri(*ctx)
                if tri:
                    transitions = [(w, c/sum(tri.values())) for w, c in tri.items()]
            
            if not transitions and context_words:
                transitions = self.tp.uni(context_words[-1])
            
            if not transitions:
                transitions = [(random.choice(self.fb), 1.0)]
        
        return transitions[:self.max_tr]
    
    def _mask_with_context(self, candidates, context_window):
        """Enhanced masking using full context window"""
        if not context_window.window:
            return np.ones(len(candidates), dtype=bool)
        
        context_words = context_window.get_context()
        seen_words = set()
        
        # Aggregate seen words from all context positions
        for word in context_words:
            seen_words.update(self.tp.bigram_counts.get(word, {}).keys())
            seen_words.add(word)
        
        return np.array([w in seen_words for w in candidates])
    
    @torch.no_grad()
    def generate(self, spikes, seed="", length=50, hidden=None, tau=0.8, 
                 adaptive_window=True, window_size=None):
        """Enhanced generation with sliding context window"""
        n_steps = length
        window_size = window_size or self.context_window_size
        
        # Initialize sliding context window
        context_window = SlidingContextWindow(max_size=window_size, min_size=2)
        
        # Cast tensors
        spikes = spikes.to(device=DEVICE, dtype=DTYPE_ACT)
        if hidden is not None:
            hidden = hidden.to(device=DEVICE, dtype=DTYPE_ACT)
        
        fg = FGParams(n_steps, tau)
        sel_probs = self.forward(spikes)
        v_mem = torch.zeros_like(spikes, device=DEVICE)
        i_syn = torch.zeros_like(spikes, device=DEVICE)
        
        # Initialize with seed
        seed_words = seed.lower().split() if seed else [random.choice(self.fb)]
        seed_words = [w for w in seed_words if w]
        
        for word in seed_words:
            context_window.add(word, importance=2.0)  # Higher importance for seed
        
        gen = list(context_window.get_context())
        
        # Completeness calculation
        high = (sel_probs > tau).float().mean().item()
        comp = torch.sigmoid(torch.tensor((high - 0.05) * 20, dtype=DTYPE_SMALL))
        
        print(f"Context window size: {window_size}, Adaptive: {adaptive_window}")
        print(f"High-prob steps {100*high:.1f}%, completeness {comp:.3f}")
        
        for i in range(n_steps):
            # Adaptive window sizing
            if adaptive_window:
                current_window_size = context_window.adaptive_resize(i, n_steps)
                context_words = context_window.get_context(current_window_size)
            else:
                context_words = context_window.get_context()
            
            # STREAM one spike row
            x = spikes[i] if i < len(spikes) else spikes[-1]
            
            # Graded LIF params
            beta = 0.8 + 0.4 * fg.s(i)
            alpha = 0.1 + 0.3 * fg.t(i)
            
            i_syn = alpha * i_syn + x
            v_mem = beta * v_mem + i_syn * torch.sigmoid(v_mem * fg.smooth) * 4
            
            # Graded threshold + spikes
            thr = 1.0 + 0.3 * math.sin(2 * math.pi * fg.s(i)) * 4
            psp = torch.sigmoid((v_mem - thr) * fg.smooth) * 4
            gumb = -torch.log(-torch.log(torch.rand_like(psp) + 1e-8) + 1e-8) * 4
            spikes_i = torch.sigmoid((torch.log(psp + 1e-8) - torch.log1p(-psp) + gumb) / 0.1)
            
            # Synthetic reset
            sf = fg.s(i)
            diff = torch.linspace(0, 4, len(x), device=DEVICE, dtype=DTYPE_ACT)
            rew = torch.abs(diff * 100 - diff * 20000)
            combined = torch.clamp(rew * (0.8 + 0.4 * sf), -2, 2)
            reset_s = torch.sigmoid(spikes_i if sf <= tau else spikes_i * (3 + rew * sf))
            v_mem = v_mem * (1 - reset_s) + combined * reset_s.unsqueeze(0)
            
            # Enhanced transition candidates using context window
            trans = self._enhanced_multi_trans(context_window)
            if not trans:
                trans = [(random.choice(self.fb), 1.0)]
            
            words, wts = zip(*trans)
            wts = np.array(wts, dtype=np.float32)
            
            # Probability calculations
            base_p = sel_probs[i].mean().item() if i < len(sel_probs) else 0.5
            hid_p = torch.sigmoid(hidden[i]).mean().item() if hidden is not None and i < len(hidden) else 1
            prob_use = hid_p if hid_p is not None else base_p
            
            # Enhanced weight modulation with context influence
            context_influence = len(context_words) / window_size  # More context = more stability
            smooth_mod = (hid_p + 0.8 * fg.s(i)) if prob_use <= tau else (0.6 + 0.3 * fg.s(i))
            wts = wts * (smooth_mod + context_influence * 0.2)
            
            # Enhanced dataset constraint using context window
            if comp < 0.7 and len(context_words) >= 2:
                mask = self._mask_with_context(words, context_window)
                wts = np.where(mask, wts, wts * 0.3)
            
            # Sample next word
            wts /= wts.sum()
            nxt = np.random.choice(words, p=wts)
            
            # Calculate importance score for the new word
            word_importance = 1.0 + context_influence * 0.5
            if nxt in self.tp.word_counts:
                # More frequent words get slightly lower importance
                freq_factor = min(1.0, 100.0 / self.tp.word_counts[nxt])
                word_importance *= freq_factor
            
            # Add to context window and generation
            context_window.add(nxt, importance=word_importance)
            gen.append(nxt)
            
            # Memory cleanup
            del spikes_i, reset_s
        
        return " ".join(gen)

# --------------------------------------------------------------------------- #
#                               Enhanced Demo                                 #
# --------------------------------------------------------------------------- #
def demo():
    model = FGNNTextGen(context_window_size=8).to(DEVICE, dtype=DTYPE_ACT)
    
    # For demo, create some sample text if no file provided
    sample_text = """
    The sliding context window enables dynamic text generation with improved coherence.
    Neural networks benefit from adaptive context management and memory optimization.
    Functionally graded parameters provide smooth transitions between generation states.
    """
    
    print("Demo mode - using sample text")
    model.tp.ingest(sample_text, max_ngram=4)
    
    B, T = 50, 100
    spikes = torch.randn(B, T, dtype=DTYPE_ACT)
    hidden = torch.randn(B, 128, dtype=DTYPE_ACT)
    
    # Demo with different window configurations
    configs = [
        {"window_size": 4, "adaptive_window": False},
        {"window_size": 8, "adaptive_window": True},
        {"window_size": 12, "adaptive_window": True}
    ]
    while True:
        seed = input("USER: ")
        for config in configs:
            print(f"\n{'='*60}")
            print(f"Configuration: {config}")
            print('='*60)
            
            txt = model.generate(
                spikes, 
                seed=seed, 
                length=500, 
                hidden=hidden,
                **config
            )
            print(f"Generated: {txt}")

if __name__ == "__main__":
    demo()
