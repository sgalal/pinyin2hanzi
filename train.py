#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
import os
import random

from model import Model

def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	# Path
	parser.add_argument('--data-train-path', default='data/ai_shell_train_sd', type=str)
	parser.add_argument('--data-dev-path', default='data/ai_shell_dev_sd', type=str)
	# Model
	parser.add_argument('--batch-size', default=32, type=int)
	parser.add_argument('--emb-dim', default=512, type=int)
	parser.add_argument('--hidden-dim', default=512, type=int)
	parser.add_argument('--n-layers', default=2, type=int)
	# Train
	parser.add_argument('--seed', default=666666, type=int)
	parser.add_argument('--n-epoch', default=100, type=int)
	return parser.parse_args()

args = get_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True  # Make the training reproducible

py_field = Field(tokenize=list, init_token='<sos>', eos_token='<eos>', batch_first=True)
han_field = Field(tokenize=list, init_token='<sos>', eos_token='<eos>', batch_first=True)

train_data = TranslationDataset(args.data_train_path, ('.pinyin', '.han'), (py_field, han_field))
valid_data = TranslationDataset(args.data_dev_path, ('.pinyin', '.han'), (py_field, han_field))

py_field.build_vocab(train_data)
han_field.build_vocab(train_data)

with open('./data/py_vocab_sd.txt', 'w') as f:
	for word in py_field.vocab.stoi:
		print(word, file=f)

with open('./data/han_vocab_sd.txt', 'w') as f:
	for word in han_field.vocab.stoi:
		print(word, file=f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, valid_loader = BucketIterator.splits((train_data, valid_data), batch_size=args.batch_size, device=device)

py_vocab_size = len(py_field.vocab.stoi)
ch_vocab_size = len(han_field.vocab.stoi)

model = Model(py_vocab_size, args.emb_dim, args.hidden_dim, ch_vocab_size, args.n_layers).to(device)


def init_weights(m):
	for name, param in m.named_parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)


def continue_train():
	import re
	res = []
	for file in os.listdir('models'):
		match = re.match(f'py2han_sd_model_epoch(\d+)val_acc(\d.\d+)', file)
		if match:
			res.append((int(match[1]), float(match[2]), os.path.join('models', file)))
	if res:
		res.sort(reverse=True)
		return res[0]

is_continue_train = continue_train()

if is_continue_train is None:
	model = model.apply(init_weights)
	epoch_, val_acc_ = -1, 0.0
else:
	epoch_, val_acc_, path_ = is_continue_train
	model.load_state_dict(torch.load(path_))


PAD_IDX = han_field.vocab.stoi['<pad>']
criterion = nn.NLLLoss(ignore_index=PAD_IDX)


tok = ['<eos>', '<unk>', '<sos>', '<pad>']

def int2str(ix):
	res = (han_field.vocab.itos[i] for i in ix)
	subst_unk = ('?' if re == '<unk>' else re for re in res)
	return ''.join(re for re in subst_unk if re not in tok)

get_weight = lambda ch: 1 if ch == 'x' else 10

def compare_pre_target(output, target, show_txt=True):
	pred = torch.argmax(output, -1)

	t_text = int2str(target[0].cpu().numpy())
	s_text = int2str(pred[0].cpu().numpy())[:len(t_text)]

	correct = sum((s == t) * get_weight(s) for s, t in zip(s_text, t_text))
	total = sum(get_weight(t) for _, t in zip(s_text, t_text))
	try:
		acc = correct / total
	except ZeroDivisionError:
		acc = 0

	if show_txt:
		print('pred:', s_text)
		print('true:', t_text)
		print('acc:', acc)

	return acc


def evaluate(model, iterator, criterion):
	model.eval()
	print('Evaluating...')

	val_acc = 0
	with torch.no_grad():
		for i, batch in enumerate(iterator):
			src = batch.src
			trg = batch.trg

			output = model(src)
			acc = compare_pre_target(output.detach(), trg.detach(), i % 64 == 0)

			output = output.contiguous().view(-1, output.shape[-1])
			trg = trg.contiguous().view(-1)

			val_acc = (val_acc * i + acc) / (i + 1)
			msg = 'val acc: {:.3}'.format(val_acc)

	print('Done')
	return val_acc

grad_clip = 1.0
optimizer = optim.Adam(model.parameters(), lr=3e-4)


def train():
	best_val_acc = val_acc_
	for epoch in range(epoch_ + 1, args.n_epoch):
		model.train()
		epoch_loss = 0
		epoch_acc = 0
		for i, batch in enumerate(train_loader):
			src = batch.src
			trg = batch.trg
			if trg.shape[0] == 0:
				continue
			optimizer.zero_grad()

			output = model(src)

			show_txt = (i % 1024 == 0)
			acc = compare_pre_target(output.detach(), trg.detach(), show_txt)

			# trg = [trg sent len, batch size]
			# output = [trg sent len, batch size, output dim]

			output = output[:].view(-1, output.shape[-1])
			trg = trg[:].view(-1)

			if trg.shape[0] == 0:
				continue

			# trg = [(trg sent len - 1) * batch size]
			# output = [(trg sent len - 1) * batch size, output dim]
			loss = criterion(output, trg)

			loss.backward()

			#torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

			optimizer.step()

			epoch_loss = (epoch_loss*i + loss.item())/(i+1)
			epoch_acc = (epoch_acc*i + acc)/(i+1)

			msg = 'loss:{:.5},acc:{:.5}'.format(epoch_loss, epoch_acc)

		val_acc = evaluate(model, valid_loader, criterion)

		optimizer.param_groups[0]['lr'] *= 0.9
		print('lr:', optimizer.param_groups[0]['lr'])

		model_path = './models/py2han_sd_model_epoch%02dval_acc%.4f.pth' % (epoch, val_acc)

		if val_acc > best_val_acc:
			print('Validation acc increased from {} to {}, saving model to {}'.format(best_val_acc, val_acc, model_path))
			best_val_acc = val_acc

		torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
	train()
