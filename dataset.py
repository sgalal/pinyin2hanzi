import os
from tokenizer import CharLevelTokenizer
import torch
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
	def __init__(self, root='./data', train=True, data_length=54, device='cpu'):
		vocab_file_x = os.path.join(root, 'vocab_x.txt')
		vocab_file_y = os.path.join(root, 'vocab_y.txt')

		target_file_x = os.path.join(root, 'train_x.txt' if train else 'test_x.txt')
		target_file_y = os.path.join(root, 'train_y.txt' if train else 'test_y.txt')

		should_initialize = not os.path.exists(vocab_file_x) or not os.path.exists(vocab_file_y)

		if should_initialize and not train:
			raise RuntimeError('You should initialize the train dataset first.')

		if should_initialize:
			# Build vocab_x
			tokenizer_x = CharLevelTokenizer(data_length=data_length)
			with open(target_file_x) as f:
				lines_x = [line.rstrip() for line in f]
			self.x = torch.tensor(tokenizer_x.build_vocab(lines_x), device=device)
			tokenizer_x.save_vocab(vocab_file_x)
			self.tokenizer_x = tokenizer_x

			# Build vocab_y
			tokenizer_y = CharLevelTokenizer(data_length=data_length)
			with open(target_file_y) as f:
				lines_y = [line.rstrip() for line in f]
			self.y = torch.tensor(tokenizer_y.build_vocab(lines_y), device=device)
			tokenizer_y.save_vocab(vocab_file_y)
			self.tokenizer_y = tokenizer_y
		else:
			# Load vocab_x
			tokenizer_x = CharLevelTokenizer(data_length=data_length)
			tokenizer_x.load_vocab(vocab_file_x)
			with open(target_file_x) as f:
				lines_x = [line.rstrip() for line in f]
			self.x = torch.tensor(tokenizer_x.stoi(lines_x), device=device)
			self.tokenizer_x = tokenizer_x

			# Load vocab_y
			tokenizer_y = CharLevelTokenizer(data_length=data_length)
			tokenizer_y.load_vocab(vocab_file_y)
			with open(target_file_y) as f:
				lines_y = [line.rstrip() for line in f]
			self.y = torch.tensor(tokenizer_y.stoi(lines_y), device=device)
			self.tokenizer_y = tokenizer_y

		assert len(self.x) == len(self.y)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
