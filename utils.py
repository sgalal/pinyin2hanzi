import torch
import torch.nn as nn

def init_weights(model: nn.Module):
	for param in model.parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)

# Use the rand function in torch package so that we can resume the rand state
def randrange(n: int):
	return int(torch.rand(1).item() * n)
