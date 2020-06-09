# -*- coding: utf-8 -*-

from model import Model
import torch
import json
import argparse

def read_vocab(file):
	with open(file) as f:
		return json.load(f)

def py_to_han(model, s, py_vocab, han_vocab_itos):
	s = '<sos> ' + s + ' <eos>'
	x = []
	for _s in s.split(' '):
		if _s in py_vocab.keys():
			x.append(py_vocab[_s])
		else:
			x.append(py_vocab['<unk>'])

	x = torch.tensor(x).cuda()
	x = torch.unsqueeze(x, 0)

	with torch.no_grad():
		model.eval()
		y = model(x)
	idx = torch.argmax(y[0].cpu(), 1)
	h = []
	for i in idx:
		h.append(han_vocab_itos[int(i)])
	return ''.join(h[1:-1])

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--py-vocab', default='./data/py_vocab_sd.txt', type=str, required=False)
	parser.add_argument('--han-vocab', default='./data/han_vocab_sd.txt', type=str, required=False)
	parser.add_argument('--model-weight', default='./models/py2han_sd_model_epoch6val_acc0.897.pth', type=str, required=False)
	return parser.parse_args()

args = get_args()

han_vocab = read_vocab(args.han_vocab)
py_vocab = read_vocab(args.py_vocab)

han_vocab_itos = {v: k for k, v in han_vocab.items()}
py_vocab_itos = {v: k for k, v in py_vocab.items()}

py_vocab_size = len(py_vocab)
ch_vocab_size = len(han_vocab)

emb_dim = 512
hidden_dim = 512
n_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sd = torch.load(args.model_weight)
model = Model(py_vocab_size, emb_dim, hidden_dim, ch_vocab_size, n_layers).to(device)
model.load_state_dict(sd)

def convert(s):
	return py_to_han(model, s, py_vocab, han_vocab_itos)
