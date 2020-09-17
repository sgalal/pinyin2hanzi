import torch
import torch.nn as nn

import config as CONFIG
from model import Model
from itos import Itos
from stoi import Stoi

# Initialize

stoi_x = Stoi('data/vocab_x.txt')
itos_y = Itos('data/vocab_y.txt')

x_vocab_size = stoi_x.vocab_size()
y_vocab_size = itos_y.vocab_size()

model = Model(x_vocab_size, CONFIG.EMB_DIM, CONFIG.HIDDEN_DIM, y_vocab_size, CONFIG.N_LAYERS)
state = torch.load(CONFIG.MODEL_PATH)
model.load_state_dict(state['state_dict'])

try:
	while True:
		x = input('> ')
		x = stoi_x(x)
		x = torch.tensor([x])
		y = model(x).permute(0, 2, 1)
		y = y[0].argmax(0).tolist()
		y = itos_y(y)
		print(y)
except EOFError:
	pass
