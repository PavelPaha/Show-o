from contextlib import nullcontext
from torch.profiler import profile, record_function, ProfilerActivity

def get_profiling_context_torch(cfg):
    prof_enabled = cfg.profiler_enabled
    prof_ctx = (nullcontext(), profile(activities=[ProfilerActivity.CPU], record_shapes=True))[prof_enabled]
    rec_ctx  = (nullcontext(), record_function("model_inference"))[prof_enabled]
    return prof_ctx, rec_ctx

# def get_profiling_context_accelerate(cfg):
