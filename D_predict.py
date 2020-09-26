from glob import glob
import torch

import config as CONFIG
from model import Model
from itos import Itos
from stoi import Stoi

# Initialize

stoi_x = Stoi('data/vocab_x.txt')
itos_y = Itos('data/vocab_y.txt')

x_vocab_size = stoi_x.vocab_size()
y_vocab_size = itos_y.vocab_size()

model = Model.load_from_checkpoint(next(iter(glob('lightning_logs/version_0/checkpoints/*.ckpt'))), map_location='cpu')

try:
	with torch.no_grad():
		while True:
			s = input('> ')
			s = stoi_x(s)
			x = torch.tensor([s])
			y = model(x)
			y = y[0].argmax(0).tolist()
			y = itos_y(y)
			print(y)
except EOFError:
	pass
