import torch

# Use the rand function in torch package so that we can resume the rand state
def randrange(n: int):
	return int(torch.rand(1).item() * n)
