import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config as CONFIG
from dataset import SentenceDataset
from itos import Itos
from model import Model
from trainutils import save, test, train

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

model = Model(x_vocab_size, CONFIG.EMB_SIZE, CONFIG.HIDDEN_SIZE, y_vocab_size, CONFIG.NUM_LAYERS, CONFIG.DROPOUT).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG.LEARNING_RATE)

# Load states

if not os.path.exists(CONFIG.MODEL_PATH):
	current_epoch = 0
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)
	model.init_weights()
else:
	state = torch.load(CONFIG.MODEL_PATH, map_location='cpu')
	current_epoch = state['epoch']
	model.load_state_dict(state['state_dict'])
	optimizer.load_state_dict(state['optimizer'])
	torch.set_rng_state(state['rng_state'])

# Start training

if __name__ == '__main__':
	while current_epoch < CONFIG.TOTAL_EPOCH:
		train(train_loader, model, criterion, optimizer, total_batch, current_epoch, itos_x, itos_y)
		current_epoch += 1
		test(test_loader, model, criterion, current_epoch, itos_x, itos_y)
		save(current_epoch, model, optimizer)
