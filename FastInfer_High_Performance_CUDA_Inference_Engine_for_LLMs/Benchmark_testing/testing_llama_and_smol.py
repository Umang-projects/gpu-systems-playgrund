import torch
import time
import matplotlib.pyplot as plt
import gc
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP

# Check Installation
try:
    import smol_kernels
    print("✅ smol_kernels loaded successfully!")
except ImportError:
    print("❌ Error: Run 'pip install .' first")
    exit()

# Define Models to Test
MODELS = {
    "SmolLM3-3B": "HuggingFaceTB/SmolLM3-3B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct"
}

# =========================================================
# 1. Define Wrappers (Injectors)
# =========================================================

# A. Fast RMSNorm
class FastRMS(torch.nn.Module):
    def __init__(self, l):
        super().__init__()
        self.w, self.e = l.weight, l.variance_epsilon
    def forward(self, x):
        return smol_kernels.rmsnorm(x, self.w, self.e)

# B. Fast SwiGLU (MLP)
def fast_mlp(self, x):
    g, u = self.gate_proj(x), self.up_proj(x)
    o = torch.empty_like(g)
    smol_kernels.swiglu(g, u, o)
    return self.down_proj(o)

# C. Fast RoPE (The Library Hack) 🧨
def fast_rope_embedding(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Replaces: transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    """
    # HF broadcasts cos/sin (1,1,Seq,Dim). We need to match q/k (Batch,Heads,Seq,Dim)
    # contiguous() is CRITICAL for C++ kernels to read memory correctly
    cos = cos.expand_as(q).contiguous()
    sin = sin.expand_as(q).contiguous()
    q = q.contiguous()
    k = k.contiguous()

    # Call Custom Kernel (In-Place update)
    smol_kernels.rope(q, cos, sin)
    smol_kernels.rope(k, cos, sin)

    return q, k

# =========================================================
# 2. Benchmark Logic
# =========================================================
def run_test(model_name, model_id):
    print(f"\n🚀 Testing {model_name} with FULL INJECTION (RMS + SwiGLU + RoPE)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cuda", 
        low_cpu_mem_usage=True
    )
    inputs = tokenizer("Artificial Intelligence is evolving because", return_tensors="pt").to("cuda")

    # --- Baseline Run ---
    print("  🐢 Running Baseline...")
    for _ in range(3): model.generate(**inputs, max_new_tokens=10) # Warmup
    
    start = time.time()
    with torch.no_grad(): 
        model.generate(**inputs, max_new_tokens=100, do_sample=False)
    torch.cuda.synchronize()
    base_tps = 100 / (time.time() - start)
    print(f"  Baseline Speed: {base_tps:.2f} TPS")

    # --- Injection Phase ---
    print("  💉 Injecting Kernels...")
    
    # 1. Inject RMSNorm & MLP
    for l in model.model.layers:
        l.input_layernorm = FastRMS(l.input_layernorm)
        l.post_attention_layernorm = FastRMS(l.post_attention_layernorm)
        l.mlp.forward = fast_mlp.__get__(l.mlp, LlamaMLP)
    model.model.norm = FastRMS(model.model.norm)

    # 2. Inject RoPE (Monkey Patching the Library Function)
    import transformers.models.llama.modeling_llama as llama_mod
    original_rope = llama_mod.apply_rotary_pos_emb # Backup original function
    llama_mod.apply_rotary_pos_emb = fast_rope_embedding # OVERRIDE with Custom Kernel

    print("  ✅ All Kernels Injected!")

    # --- Optimized Run ---
    print("  🚀 Running Optimized...")
    for _ in range(3): model.generate(**inputs, max_new_tokens=10) # Warmup
    
    start = time.time()
    with torch.no_grad(): 
        model.generate(**inputs, max_new_tokens=100, do_sample=False)
    torch.cuda.synchronize()
    opt_tps = 100 / (time.time() - start)
    print(f"  Custom Speed:   {opt_tps:.2f} TPS")

    # --- Restore & Cleanup ---
    llama_mod.apply_rotary_pos_emb = original_rope # RESTORE Original Function for next model
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return base_tps, opt_tps

# =========================================================
# 3. Execute & Plot
# =========================================================
results = {}
for name, m_id in MODELS.items():
    results[name] = run_test(name, m_id)

# Plotting
print("\n📊 Generating Graph...")
names = list(results.keys())
base = [results[n][0] for n in names]
opt = [results[n][1] for n in names]

x = range(len(names))
plt.figure(figsize=(8, 6))
# Bars
plt.bar([i-0.2 for i in x], base, 0.4, label='Baseline', color='#ff9999', edgecolor='black')
plt.bar([i+0.2 for i in x], opt, 0.4, label='RMS+SwiGLU+RoPE', color='#00E676', edgecolor='black')

# Labels
for i in x:
    plt.text(i-0.2, base[i]+0.5, f"{base[i]:.1f}", ha='center')
    plt.text(i+0.2, opt[i]+0.5, f"{opt[i]:.1f}", ha='center', fontweight='bold')
    speedup = ((opt[i]-base[i])/base[i])*100
    plt.text(i, opt[i]+3, f"+{speedup:.1f}%", ha='center', color='darkgreen', fontweight='bold', fontsize=12)

plt.xticks(x, names)
plt.ylabel("Tokens / Sec")
plt.title("Full Kernel Injection Speedup (Tesla T4 FP16)")
plt.legend()
plt.ylim(0, max(opt)+5)
plt.savefig("benchmark_full_injection.png")
print("✅ Graph saved as 'benchmark_full_injection.png'")