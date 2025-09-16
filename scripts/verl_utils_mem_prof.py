import time, torch, json, os, datetime
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

def _nvml_used(dev):
    if not _NVML_AVAILABLE: 
        return -1
    h = pynvml.nvmlDeviceGetHandleByIndex(int(dev))
    return pynvml.nvmlDeviceGetMemoryInfo(h).used

def _log(p):
    ts = datetime.datetime.now().strftime("%F %T")
    os.makedirs("/root/autodl-tmp/LLM_RL_basic/outputs", exist_ok=True)
    with open("/root/autodl-tmp/LLM_RL_basic/outputs/ppo_trace.txt", "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {json.dumps(p, ensure_ascii=False)}\n")
        f.flush()
        os.fsync(f.fileno())

class StageMem:
    def __init__(self, tag: str, dev: int | None = None):
        self.tag = tag
        self.dev = dev
        self.use_torch = False   # 标记是否能用 torch.cuda

    def __enter__(self):
        if torch.cuda.is_available():
            if self.dev is None:
                self.dev = torch.cuda.current_device()
            self.use_torch = True
            torch.cuda.synchronize(self.dev)
            self.t0 = time.time()
            self.base_alloc = torch.cuda.memory_allocated(self.dev)
            self.base_resv  = torch.cuda.memory_reserved(self.dev)
            self.base_nvml  = _nvml_used(self.dev)
            torch.cuda.reset_peak_memory_stats(self.dev)
        else:
            # 没 GPU，fallback: 只用 NVML 看总量
            self.dev = 0 if self.dev is None else self.dev
            self.use_torch = False
            self.t0 = time.time()
            self.base_alloc = -1
            self.base_resv  = -1
            self.base_nvml  = _nvml_used(self.dev)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.t1 = time.time()
        k = f"dev{self.dev}/{self.tag}"
        if self.use_torch:
            torch.cuda.synchronize(self.dev)
            self.peak_alloc = torch.cuda.max_memory_allocated(self.dev)
            self.peak_resv  = torch.cuda.max_memory_reserved(self.dev)
            self.end_alloc  = torch.cuda.memory_allocated(self.dev)
            self.end_resv   = torch.cuda.memory_reserved(self.dev)
            self.end_nvml   = _nvml_used(self.dev)
            _log({
                f"dev/tag": k,
                f"time_s": round(self.t1 - self.t0, 4),
                f"peak_alloc_GB": round(self.peak_alloc / (1024**3), 3),
                f"peak_resv_GB": round(self.peak_resv / (1024**3), 3),
                f"base_alloc_GB": round(self.base_alloc / (1024**3), 3), 
                f"delta_alloc_GB": round((self.end_alloc - self.base_alloc) / (1024**3), 3),
                f"delta_resv_GB": round((self.end_resv - self.base_resv) / (1024**3), 3),
                f"nvml_base_GB": round(self.base_nvml / (1024**3), 3),
                f"nvml_end_GB": round(self.end_nvml / (1024**3), 3),
                f"nvml_delta_GB": round((self.end_nvml - self.base_nvml) / (1024**3), 3)
                                    if (self.base_nvml >= 0 and self.end_nvml >= 0) else -1,
            })
        else:
            # 没 torch.cuda 的情况，只能看 NVML
            self.peak_alloc = -1
            self.peak_resv  = -1
            self.end_alloc  = -1
            self.end_resv   = -1
            self.end_nvml   = _nvml_used(self.dev)
            _log({
                f"dev/tag": k,
                f"time_s": round(self.t1 - self.t0, 4),
                f"nvml_base_GB": round(self.base_nvml / (1024**3), 3),
                f"nvml_end_GB": round(self.end_nvml / (1024**3), 3),
                f"nvml_delta_GB": round((self.end_nvml - self.base_nvml) / (1024**3), 3)
                                    if (self.base_nvml >= 0 and self.end_nvml >= 0) else -1,
            })

'''
要加在 worker 的函数内部 (with StageMem('xxx') as m:)，rayppotrainer看不了GPU。
先给模型参数bf16 + 梯度f32常驻10GB，通信桶+其它约9GB。共19GB随TP减少。还有额外1GB。
再给vllm reserve。KV cache B=1, l=1024 0.12GB，随TP减少。
计算log_prob顶峰显存为11.5GB (1.7B, tp=2, l<14000, log_prob_micro=2)。包含计算entropy，为 B*l*vocabsize*dtype / TP。
第一次优化常驻f32主参数。
优化顶峰显存（不含f32）是15.8GB (1.7B, tp=2, l<14000, micro=1)。大概是6.88*4.6 / TP。
每个step会额外增加0.075GB。
'''