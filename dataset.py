import torch
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
	def __init__(self, train=True, device='cpu'):
		if train:
			self.x = torch.load('data/tokens_train_x.pth', map_location=device)
			self.y = torch.load('data/tokens_train_y.pth', map_location=device)
		else:
			self.x = torch.load('data/tokens_test_x.pth', map_location=device)
			self.y = torch.load('data/tokens_test_y.pth', map_location=device)
		assert len(self.x) == len(self.y)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
