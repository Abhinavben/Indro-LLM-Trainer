import os
import math
import time
import copy
import contextlib
import random
import threading
from queue import Queue
import zstandard as zstd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, ShardedStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from huggingface_hub import hf_hub_download, HfApi
import functools
import wandb
import pynvml

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# ==========================================
# 1. DEEPMIND-CLASS HYPERPARAMETERS
# ==========================================
CONFIG = {
    "hf_repo_id": "Indro-ai/Indro-3B-Corpus",
    "hf_model_repo": "Indro-ai/Indro-3B-Omni",
    "vocab_size": 65024,      
    "max_seq_length": 2048,   
    "batch_size": 8,          
    "grad_accum_steps": 4,    
    "base_lr": 3e-4, 
    "warmup_steps": 2000,     
    "eval_interval": 500,     
    "save_interval": 1000,    
    "dropout_start": 0.1,     
    "dropout_end": 0.0,
    "ema_decay": 0.999,       
    "max_ckpts": 3,           
    "max_steps": 100000,      
    "weight_decay": 0.1,
    "bos_token": 1,           
    "eos_token": 2,
    "z_loss_max_weight": 1e-4, # 👑 NEW: PaLM-style Logit Stabilization
    "wandb_project": "Indro-3B-Singularity-V5",
    "seed": 42
}

torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
torch.backends.cudnn.deterministic = True

# ==========================================
# 2. AUTONOMOUS METRICS & CONTROLLERS
# ==========================================
class TrainingMonitor:
    def __init__(self):
        self.loss_history = []
        self.skipped_steps = 0 # 👑 FIX: Track gradient explosions
    
    def update(self, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) > 100: self.loss_history.pop(0)

    def detect_anomaly(self, current_loss):
        if len(self.loss_history) < 50: return False
        mean = np.mean(self.loss_history[-50:])
        std = np.std(self.loss_history[-50:])
        return current_loss > (mean + 3 * std)

class OmniLR:
    def __init__(self, base_lr, warmup_steps, max_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def update(self, step):
        if step < self.warmup_steps:
            return self.base_lr * (step / max(1, self.warmup_steps))
        decay_ratio = (step - self.warmup_steps) / max(1, (self.max_steps - self.warmup_steps))
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

def get_curriculum_seq_length(step):
    if step < 5000: return 512
    if step < 15000: return 1024
    return CONFIG["max_seq_length"]

def get_annealed_dropout(step):
    ratio = min(1.0, step / CONFIG["max_steps"])
    return CONFIG["dropout_start"] - (ratio * (CONFIG["dropout_start"] - CONFIG["dropout_end"]))

def get_z_loss_weight(step):
    """👑 FIX: Safely ramps up Z-Loss only after the model is stable"""
    if step < 5000: return 0.0
    ratio = min(1.0, (step - 5000) / 10000.0)
    return CONFIG["z_loss_max_weight"] * ratio

# ==========================================
# 3. SHANNON ENTROPY DATA PIPELINE
# ==========================================
def calculate_shannon_entropy(tokens):
    """👑 NEW: Mathematical Data Quality Filter"""
    counts = np.bincount(tokens)
    probs = counts[counts > 0] / len(tokens)
    return -np.sum(probs * np.log2(probs))

class OmniCloudDataset(IterableDataset):
    def __init__(self, repo_id, max_seq_length, rank, world_size, seed, is_val=False, global_step_tensor=None):
        self.api = HfApi()
        self.max_seq_length = max_seq_length
        self.rank, self.world_size = rank, world_size
        self.global_step_tensor = global_step_tensor 
        
        all_files = sorted([f for f in self.api.list_repo_files(repo_id, repo_type="dataset") if f.endswith('.bin.zst')])
        random.Random(seed).shuffle(all_files)
        val_split = max(1, len(all_files) // 100)
        self.files = all_files[:val_split] if is_val else all_files[val_split:]
        self.cache_dir = "./hf_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        my_files = self.files[self.rank :: self.world_size][worker_id :: num_workers]

        dctx = zstd.ZstdDecompressor()
        buffer_tokens = []
        buffer_doc_ids = [] 
        doc_id_counter = 1
        
        for file_name in my_files:
            local_path = hf_hub_download(repo_id=CONFIG["hf_repo_id"], filename=file_name, repo_type="dataset", cache_dir=self.cache_dir)
            with open(local_path, 'rb') as f:
                with dctx.stream_reader(f) as reader:
                    while True:
                        raw_bytes = reader.read(4096)
                        if len(raw_bytes) == 0: break
                        tokens = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.int64).tolist()
                        
                        # 👑 FIX: Entropy + Length Filtering
                        if len(tokens) < 50: continue 
                        if calculate_shannon_entropy(tokens) < 3.5: continue # Rejects repeated spam instantly
                        
                        doc = [CONFIG["bos_token"]] + tokens + [CONFIG["eos_token"]]
                        doc_ids = [doc_id_counter] * len(doc)
                        
                        buffer_tokens.extend(doc)
                        buffer_doc_ids.extend(doc_ids)
                        doc_id_counter += 1
                        
                        current_step = self.global_step_tensor[0].item() if self.global_step_tensor is not None else 0
                        current_seq_len = get_curriculum_seq_length(current_step)
                        chunk_size = current_seq_len + 1 
                        
                        while len(buffer_tokens) >= chunk_size:
                            chunk_toks = buffer_tokens[:chunk_size]
                            chunk_docs = buffer_doc_ids[:chunk_size]
                            
                            buffer_tokens = buffer_tokens[chunk_size:]
                            buffer_doc_ids = buffer_doc_ids[chunk_size:]
                            
                            x = torch.tensor(chunk_toks[:-1], dtype=torch.long)
                            y = torch.tensor(chunk_toks[1:], dtype=torch.long)
                            docs = torch.tensor(chunk_docs[:-1], dtype=torch.long)
                            
                            yield x, y, docs

# ==========================================
# 4. ARCHITECTURE: FLASH ATTN V2 + SELECTIVE CHECKPOINTING
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x): 
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    return torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq), torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

class IndroAttention(nn.Module):
    def __init__(self, d_model=3072, n_heads=24):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis, dropout_p=0.0, doc_ids=None):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        attn_mask = None
        # 👑 FIX: Memory Explosion Prevention. Only apply dense mask if T <= 1024
        if doc_ids is not None and T <= 1024:
            attn_mask = (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)).unsqueeze(1) 
            causal_mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool)).view(1, 1, T, T)
            attn_mask = attn_mask & causal_mask
        
        if HAS_FLASH_ATTN and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16) and attn_mask is None:
            y = flash_attn_func(q, k, v, dropout_p=dropout_p if self.training else 0.0, causal=True)
            y = y.contiguous().view(B, T, C)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None), dropout_p=dropout_p if self.training else 0.0)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            
        return F.dropout(self.wo(y), p=dropout_p, training=self.training)

class IndroTransformerBlock(nn.Module):
    def __init__(self, d_model=3072, n_heads=24):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attention = IndroAttention(d_model, n_heads)
        self.w1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.w2 = nn.Linear(4 * d_model, d_model, bias=False)
        self.w3 = nn.Linear(d_model, 4 * d_model, bias=False)

    def forward(self, x, freqs_cis, dropout_p, doc_ids=None):
        h = x + self.attention(self.attn_norm(x), freqs_cis, dropout_p, doc_ids)
        ffn_in = self.ffn_norm(h)
        ffn_out = self.w2(F.silu(self.w1(ffn_in)) * self.w3(ffn_in))
        return h + F.dropout(ffn_out, p=dropout_p, training=self.training)

class Indro3B(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(CONFIG["vocab_size"], 3072)
        self.layers = nn.ModuleList([IndroTransformerBlock() for _ in range(32)])
        self.norm = RMSNorm(3072)
        self.output = nn.Linear(3072, CONFIG["vocab_size"], bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.freqs_cis = precompute_freqs_cis(3072 // 24, CONFIG["max_seq_length"] * 2)

    def forward(self, idx, dropout_p=0.0, doc_ids=None):
        h = self.tok_embeddings(idx)
        freqs_cis = self.freqs_cis[:idx.shape[1]].to(idx.device)
        for i, layer in enumerate(self.layers):
            # 👑 NEW: Selective Activation Checkpointing (Saves VRAM + Speeds up compute)
            if self.training and i % 2 == 0:
                h = checkpoint(layer, h, freqs_cis, dropout_p, doc_ids, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, dropout_p, doc_ids)
        return self.output(self.norm(h))

# ==========================================
# 5. LAB EVALUATION & SAFETY MECHANISMS
# ==========================================
def get_parameter_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.ndim <= 1 or "norm" in name: no_decay.append(param)
        else: decay.append(param)
    return [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]

@torch.no_grad()
def estimate_loss(ema_model, dataloader, device, world_size, eval_iters=50):
    ema_model = ema_model.to(device, non_blocking=True)
    ema_model.eval()
    losses, accuracies = [], []
    data_iter = iter(dataloader)
    
    for _ in range(eval_iters):
        try: x, y, docs = next(data_iter)
        except StopIteration: break
        x, y, docs = x.to(device, non_blocking=True), y.to(device, non_blocking=True), docs.to(device, non_blocking=True)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = ema_model(x, dropout_p=0.0, doc_ids=docs)
            loss = F.cross_entropy(logits.view(-1, CONFIG["vocab_size"]), y.view(-1))
            acc = (logits.argmax(dim=-1) == y).float().mean()
            
        losses.append(loss.item())
        accuracies.append(acc.item())
        
    ema_model = ema_model.cpu()
    torch.cuda.empty_cache()
    if len(losses) == 0: return float("inf"), 0.0
    
    local_loss = torch.tensor(np.mean(losses)).to(device)
    local_acc = torch.tensor(np.mean(accuracies)).to(device)
    dist.all_reduce(local_loss) 
    dist.all_reduce(local_acc)
    return (local_loss.item() / world_size), (local_acc.item() / world_size)

def save_sharded_checkpoint(model, optimizer, step, rank, save_type="latest"):
    save_policy = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, save_policy):
        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
        dist.barrier()
        torch.save(state_dict, f"indro_{save_type}_rank{rank}.pt")
    if rank == 0: 
        print(f"💾 SHARDED CHECKPOINT SECURED: indro_{save_type}")

# ==========================================
# 6. APOTHEOSIS TRAINING LOOP
# ==========================================
def train():
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0: 
        wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
        pynvml.nvmlInit()

    raw_model = Indro3B()
    ema_model = copy.deepcopy(raw_model).float().cpu() 
    
    raw_model = raw_model.to(device)
    auto_wrap = functools.partial(size_based_auto_wrap_policy, min_num_params=50000)
    model = FSDP(raw_model, auto_wrap_policy=auto_wrap, device_id=rank)
    
    param_groups = get_parameter_groups(model, CONFIG["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups, lr=CONFIG["base_lr"], betas=(0.9, 0.95), eps=1e-8, fused=True)
    scaler = torch.cuda.amp.GradScaler()
    
    # 👑 FIX: Global Sync Tensor setup correctly
    global_step = torch.tensor([0], dtype=torch.long, device=device)
    dist.barrier()
    if os.path.exists(f"indro_latest_rank{rank}.pt"):
        ckpt = torch.load(f"indro_latest_rank{rank}.pt", map_location="cpu")
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
        global_step[0] = ckpt["step"]
        if rank == 0: print(f"🔄 RESUMING FROM CRASH | Step {global_step[0].item()}")
    dist.barrier()

    train_data = OmniCloudDataset(CONFIG["hf_repo_id"], CONFIG["max_seq_length"], rank, world_size, CONFIG["seed"], global_step_tensor=global_step)
    val_data = OmniCloudDataset(CONFIG["hf_repo_id"], CONFIG["max_seq_length"], rank, world_size, CONFIG["seed"], is_val=True, global_step_tensor=global_step)
    
    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"], num_workers=1, pin_memory=True)
    data_iter = iter(train_loader)

    monitor = TrainingMonitor()
    lr_controller = OmniLR(CONFIG["base_lr"], CONFIG["warmup_steps"], CONFIG["max_steps"])
    
    model.train()
    running_loss, best_val_loss = 0.0, float('inf')
    t0 = time.time()

    while global_step[0].item() < CONFIG["max_steps"]:
        step_val = global_step[0].item()
        lr = lr_controller.update(step_val)
        current_dropout = get_annealed_dropout(step_val)
        current_seq_len = get_curriculum_seq_length(step_val)
        current_z_weight = get_z_loss_weight(step_val) # 👑 NEW: Z-Loss Scheduler
        
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        accum_loss, accum_acc = 0, 0
        grad_norm_sq = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(CONFIG["grad_accum_steps"]):
            try: x, y, docs = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y, docs = next(data_iter)
            
            x, y, docs = x.to(device, non_blocking=True), y.to(device, non_blocking=True), docs.to(device, non_blocking=True)
            is_last = (micro_step == CONFIG["grad_accum_steps"] - 1)
            
            with model.no_sync() if not is_last else contextlib.nullcontext():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(x, dropout_p=current_dropout, doc_ids=docs)
                    
                    # 👑 FIX: PaLM-style Z-Loss implementation to prevent logit explosion
                    cross_entropy = F.cross_entropy(logits.view(-1, CONFIG["vocab_size"]), y.view(-1))
                    z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean() * current_z_weight
                    loss = (cross_entropy + z_loss) / CONFIG["grad_accum_steps"]
                    
                    acc = (logits.argmax(dim=-1) == y).float().mean() / CONFIG["grad_accum_steps"]
                    
                scaler.scale(loss).backward()
                accum_loss += loss.item()
                accum_acc += acc.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 👑 FIX: WandB Tracking for gradient explosions
        if not math.isfinite(accum_loss) or grad_norm > 10.0 or torch.isnan(grad_norm):
            if rank == 0: 
                print(f"💀 CRITICAL: Gradient {grad_norm:.2f}. Skipping Step to save weights.")
                monitor.skipped_steps += 1
                wandb.log({"system/skipped_steps": monitor.skipped_steps}, step=step_val)
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.step(optimizer)
        scaler.update()
        monitor.update(accum_loss)
        running_loss = 0.9 * running_loss + 0.1 * accum_loss if step_val > 0 else accum_loss
        
        for p in model.parameters():
            if p.grad is not None: grad_norm_sq += (p.grad.norm() ** 2).item()
        gns = grad_norm_sq / (accum_loss + 1e-8)
        
        # 👑 FIX: Safe EMA with `summon_full_params` offloaded to CPU (Executes every 50 steps to preserve speed)
        if step_val % 50 == 0:
            with FSDP.summon_full_params(model, writeback=False, offload_to_cpu=True):
                with torch.no_grad():
                    model_params = [p.data for p in model.parameters()]
                    ema_params = list(ema_model.parameters())
                    torch._foreach_mul_(ema_params, CONFIG["ema_decay"])
                    torch._foreach_add_(ema_params, model_params, alpha=1 - CONFIG["ema_decay"])

        # TELEMETRY
        if step_val % 10 == 0 and rank == 0:
            dt = time.time() - t0
            tok_sec = (CONFIG["batch_size"] * CONFIG["grad_accum_steps"]
                    
                    acc = (logits.argmax(dim=-1) == y).float().mean() / CONFIG["grad_accum_steps"]
                    
                scaler.scale(loss).backward()
                accum_loss += loss.item()
                accum_acc += acc.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 👑 FIX: WandB Tracking for gradient explosions
        if not math.isfinite(accum_loss) or grad_norm > 10.0 or torch.isnan(grad_norm):
            if rank == 0: 
                print(f"💀 CRITICAL: Gradient {grad_norm:.2f}. Skipping Step to save weights.")
                monitor.skipped_steps += 1
                wandb.log({"system/skipped_steps": monitor.skipped_steps}, step=step_val)
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.step(optimizer)
        scaler.update()
        monitor.update(accum_loss)
        running_loss = 0.9 * running_loss + 0.1 * accum_loss if step_val > 0 else accum_loss
        
        for p in model.parameters():
            if p.grad is not None: grad_norm_sq += (p.grad.norm() ** 2).item()
        gns = grad_norm_sq / (accum_loss + 1e-8)
        
        # 👑 FIX: Safe EMA with `summon_full_params` offloaded to CPU (Executes every 50 steps to preserve speed)
        if step_val % 50 == 0:
            with FSDP.summon_full_params(model, writeback=False, offload_to_cpu=True):
                with torch.no_grad():
                    model_params = [p.data for p in model.parameters()]
                    ema_params = list(ema_model.parameters())
                    torch._foreach_mul_(ema_params, CONFIG["ema_decay"])
                    torch._foreach_add_(ema_params, model_params, alpha=1 - CONFIG["ema_decay"])

        # TELEMETRY
        if step_val % 10 == 0 and rank == 0:
            dt = time.time() - t0
            tok_sec = (CONFIG["batch_size"] * CONFIG["grad_accum_steps"] * current_seq_len * world_size) / dt
            handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            perplexity = math.exp(min(accum_loss, 20))
            
            print(f"Step {step_val:5d} | Seq: {current_seq_len} | Loss: {accum_loss:.4f} | Acc: {accum_acc*100:.1f}% | Z-Loss: {current_z_weight:.6f} | GPU: {gpu_util}%")
            wandb.log({
                "train/loss": accum_loss, "train/accuracy": accum_acc, "train/lr": lr, 
                "train/grad_norm": grad_norm, "train/dropout": current_dropout, "train/seq_len": current_seq_len,
                "train/z_loss_weight": current_z_weight, "stats/gns": gns, "stats/perplexity": perplexity,
                "system/tok_sec": tok_sec, "system/gpu_util": gpu_util
            }, step=step_val)
            t0 = time.time()

        if step_val % CONFIG["eval_interval"] == 0 and step_val > 0:
            val_loss, val_acc = estimate_loss(ema_model, val_loader, device, world_size)
            if rank == 0:
                print(f"🔍 VAL LOSS (EMA): {val_loss:.4f} | Acc: {val_acc*100:.1f}%")
                wandb.log({"val/loss": val_loss, "val/accuracy": val_acc}, step=step_val)
                if val_loss > best_val_loss * 1.3:
                    print("🛑 CATASTROPHIC OVERFITTING. Halting.")
                    break
                best_val_loss = min(best_val_loss, val_loss)

        if step_val % CONFIG["save_interval"] == 0 and step_val > 0:
            save_sharded_checkpoint(model, optimizer, step_val, rank, "latest")

        # 👑 FIX: Global curriculum step synchronization across all GPUs
        if rank == 0:
            global_step[0] += 1
        dist.broadcast(global_step, src=0)

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
