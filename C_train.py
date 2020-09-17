import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config as CONFIG
from dataset import SentenceDataset
from model import Model
from itos import Itos

# Initialize

torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = SentenceDataset(train=True, device=device)
test_set = SentenceDataset(train=False, device=device)

train_loader = DataLoader(train_set, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

total_batch = len(train_loader)

itos_x = Itos('data/vocab_x.txt')
itos_y = Itos('data/vocab_y.txt')

x_vocab_size = itos_x.vocab_size()
y_vocab_size = itos_y.vocab_size()

model = Model(x_vocab_size, CONFIG.EMB_DIM, CONFIG.HIDDEN_DIM, y_vocab_size, CONFIG.N_LAYERS).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR)

# Load states

if not os.path.exists(CONFIG.MODEL_PATH):
	current_epoch = 0
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)
else:
	state = torch.load(CONFIG.MODEL_PATH, map_location='cpu')
	current_epoch = state['epoch']
	model.load_state_dict(state['state_dict'])
	optimizer.load_state_dict(state['optimizer'])
	torch.set_rng_state(state['rng_state'])

# Use the rand function in torch package so that we can resume the rand state
randrange = lambda n: int(torch.rand(1).item() * n)

# Utilities

def show_sample_result(x, y, y_hat):
	rand_idx = randrange(y.shape[0])
	sample_x = x[rand_idx].tolist()
	sample_y = y[rand_idx].tolist()
	sample_y_hat = y_hat[rand_idx].argmax(0).tolist()
	print('SI:', itos_x(sample_x))  # sample input
	print('EO:', itos_y(sample_y))  # expected output
	print('MO:', itos_y(sample_y_hat))  # model output

# Training process

def train():
	total_loss = 0
	for current_batch, (x, y) in enumerate(train_loader):
		y_hat = model(x)
		loss = criterion(y_hat, y)
		current_loss = loss.item()
		total_loss += current_loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if current_batch % 10 == 9:
			print('Epoch', current_epoch, 'batch %d/%d:' % (current_batch, total_batch), 'loss', current_loss)
			show_sample_result(x, y, y_hat)
	print('Epoch', current_epoch, 'train total loss:', total_loss)

def test():
	total_loss = 0
	with torch.no_grad():
		for x, y in test_loader:
			y_hat = model(x)
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
	torch.save(state, CONFIG.MODEL_PATH)
	print('Saved epoch', current_epoch)

# Start training

if __name__ == '__main__':
	while current_epoch < CONFIG.TOTAL_EPOCH:
		train()
		current_epoch += 1
		test()
		save()
