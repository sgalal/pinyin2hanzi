import torch
import torch.nn as nn

import config as CONFIG
from model import Model
from si import SIHelper
from itoshelper import ItosHelper

# Initialize

isx = ItosHelper('data/vocab_x.txt')
siy = SIHelper('data/vocab_y.txt')

x_vocab_size = isx.vocab_size()
y_vocab_size = siy.vocab_size()

model = Model(x_vocab_size, CONFIG.EMB_DIM, CONFIG.HIDDEN_DIM, y_vocab_size, CONFIG.N_LAYERS)
state = torch.load('data/model.pth')
model.load_state_dict(state['state_dict'])

try:
	while True:
		x = input('> ')
		x = isx.stoi(x)
		x = torch.tensor([x])
		y = model(x).permute(0, 2, 1)
		y = y[0].argmax(0).tolist()
		y = siy.itos(y)
		print(y)
except EOFError:
	pass
