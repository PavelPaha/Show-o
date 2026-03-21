import torch

# run this with `export PYTORCH_JIT_LOG_LEVEL=">>profiling_graph_executor_impl" && python test_model.py `
@torch.jit.script
def model(x, y, z, w):
    return (x + y) * z - w

device = 'cuda:0'
a = torch.rand(2, 2, device=device)
b = torch.rand(2, 2, device=device)
c = torch.rand(2, 2, device=device)
d = torch.rand(2, 2, device=device)

_ = model(a, b, c, d)
_ = model(a, b, c, d)