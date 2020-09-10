import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SentenceDataset
from model import Model

# Hyperparameters

emb_dim = 512
hidden_dim = 512
n_layers = 2

batch_size = 480
total_epoch = 16
lr = 8e-4

data_length = 54

model_save_path = 'data/model.pth'

# Initialize

torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = SentenceDataset(train=True, root='data', data_length=data_length, device=device)
test_set = SentenceDataset(train=False, root='data', data_length=data_length, device=device)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

x_vocab_size = train_set.tokenizer_x.vocab_size()
y_vocab_size = train_set.tokenizer_y.vocab_size()

model = Model(x_vocab_size, emb_dim, hidden_dim, y_vocab_size, n_layers).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Load states

if not os.path.exists(model_save_path):
	current_epoch = 0
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)
else:
	state = torch.load(model_save_path)
	current_epoch = state['epoch']
	model.load_state_dict(state['state_dict'])
	optimizer.load_state_dict(state['optimizer'])
	torch.set_rng_state(state['rng_state'])

# Use the rand function in torch package so that we can resume the rand state
randrange = lambda n: int(torch.rand(1).item() * n)

# Utilities

def remove_trailing_pad(lst):
	while lst and lst[-1] == train_set.tokenizer_x.TOK_PAD:
		lst.pop()
	return lst

def show_sample_result(x, y, y_hat):
	rand_idx = randrange(y.shape[0])
	sample_x = remove_trailing_pad(x[rand_idx].tolist())
	sample_y = remove_trailing_pad(y[rand_idx].tolist())
	sample_y_hat = remove_trailing_pad(y_hat[rand_idx].argmax(0).tolist())
	print('Sample input:', train_set.tokenizer_x.itos([sample_x])[0])
	print('Expected output:', train_set.tokenizer_y.itos([sample_y])[0])
	print('Model output:', train_set.tokenizer_y.itos([sample_y_hat])[0])

# Training process

def train():
	total_loss = 0
	for current_batch, (x, y) in enumerate(train_loader):
		y_hat = model(x).permute(0, 2, 1)
		loss = criterion(y_hat, y)
		current_loss = loss.item()
		total_loss += current_loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if current_batch % 100 == 99:
			print('Epoch', current_epoch, 'train batch', current_batch, 'loss:', current_loss)
			show_sample_result(x, y, y_hat)
	print('Epoch', current_epoch, 'train total loss:', total_loss)

def test():
	total_loss = 0
	with torch.no_grad():
		for x, y in test_loader:
			y_hat = model(x).permute(0, 2, 1)
			loss = criterion(y_hat, y)
			total_loss += loss.item()
		print('Epoch', current_epoch, 'test loss:', total_loss)
		show_sample_result(x, y, y_hat)

def save():
	state = {
		'epoch': current_epoch,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'rng_state': torch.get_rng_state(),
	}
	torch.save(state, model_save_path)
	print('Saved epoch', current_epoch)

# Start training

if __name__ == '__main__':
	while current_epoch < total_epoch:
		train()
		current_epoch += 1
		test()
		save()
