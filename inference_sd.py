# -*- coding: utf-8 -*-

from model import Model
import torch
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--pinyin-vocab', default='./data/py_vocab_sd.txt', type=str)
	parser.add_argument('--han-vocab', default='./data/han_vocab_sd.txt', type=str)
	parser.add_argument('--model-weight', default='./models/py2han_sd_model_epoch6val_acc0.897.pth', type=str)
	return parser.parse_args()

args = get_args()

with open(args.han_vocab) as f:
	han_vocab_itos = [line.rstrip() for line in f]

with open(args.pinyin_vocab) as f:
	pinyin_vocab_itos = [line.rstrip() for line in f]

han_vocab = {k: i for i, k in enumerate(han_vocab_itos)}
pinyin_vocab = {k: i for i, k in enumerate(pinyin_vocab_itos)}

han_vocab_size = len(han_vocab)
pinyin_vocab_size = len(pinyin_vocab)

emb_dim = 512
hidden_dim = 512
n_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(args.model_weight)
model = Model(pinyin_vocab_size, emb_dim, hidden_dim, han_vocab_size, n_layers).to(device)
model.load_state_dict(checkpoint)

def pinyin2int(s):
	x = [pinyin_vocab['<sos>']]
	for ch in s:
		res = pinyin_vocab.get(ch)
		if res:
			x.append(res)
		else:
			x.append(pinyin_vocab['<unk>'])
	x.append(pinyin_vocab['<eos>'])
	return x

def convert(s):
	x = pinyin2int(s)

	x = torch.tensor(x).cuda()
	x = torch.unsqueeze(x, 0)

	with torch.no_grad():
		model.eval()
		y = model(x)
	idx = torch.argmax(y[0].cpu(), 1)
	h = []
	for i in idx[1:-1]:
		h.append(han_vocab_itos[int(i)])
	return ''.join(h)
