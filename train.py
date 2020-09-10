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

batch_size = 32
total_epoch = 128
lr = 0.0008

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

# Training process

def train():
	total_loss = 0
	for batch_idx, (x, y) in enumerate(train_loader):
		y_hat = model(x).permute(0, 2, 1)
		loss = criterion(y_hat, y)
		total_loss += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print('Epoch', current_epoch, 'train loss:', total_loss)

def test():
	total_loss = 0
	with torch.no_grad():
		for x, y in test_loader:
			y_hat = model(x).permute(0, 2, 1)
			loss = criterion(y_hat, y)
			total_loss += loss.item()
	print('Epoch', current_epoch, 'test loss:', total_loss)

def visualize():
	with torch.no_grad():
		x, y = next(iter(test_loader))
		y_hat = model(x).permute(0, 2, 1)
		sample_idx = randrange(y.shape[0])
	print('Sample input:', train_set.tokenizer_x.itos([x[sample_idx]])[0])
	print('Expected output:', train_set.tokenizer_y.itos([y_hat[sample_idx].argmax(0)])[0])
	print('Model output:', train_set.tokenizer_y.itos([y[sample_idx]])[0])

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
		visualize()
		save()
