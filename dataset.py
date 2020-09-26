import torch
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
	def __init__(self):
		self.x = torch.load('data/tokens_x.pth', map_location='cpu')
		self.y = torch.load('data/tokens_y.pth', map_location='cpu')
		assert len(self.x) == len(self.y)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
