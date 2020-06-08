#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchtext
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator

import random
import math
import os
import time

import numpy as np
import json

SEED = 666666
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  # to make the training reproducible

from model import Model

def tokenize_pinyin(text):
	return text.split(' ')

def tokenize_hanzi(text):
	return list(text)

py_field = Field(tokenize=tokenize_pinyin, init_token='<sos>', eos_token='<eos>', batch_first=True)
han_field = Field(tokenize=tokenize_hanzi, init_token='<sos>', eos_token='<eos>', batch_first=True)

train_data = TranslationDataset('./data/ai_shell_train_sd', ('.pinyin', '.han'), (py_field, han_field))
valid_data = TranslationDataset('./data/ai_shell_dev_sd', ('.pinyin', '.han'), (py_field, han_field))

py_field.build_vocab(train_data, min_freq=2)
han_field.build_vocab(train_data, min_freq=2)

with open('./data/py_vocab_sd.txt','w') as f:
	json.dump(dict(py_field.vocab.stoi), f, ensure_ascii=False)

with open('./data/han_vocab_sd.txt','w') as f:
	json.dump(dict(han_field.vocab.stoi), f, ensure_ascii=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
train_loader, valid_loader = BucketIterator.splits((train_data, valid_data), batch_size=batch_size, device=device)

py_vocab_size = len(py_field.vocab.stoi)
ch_vocab_size = len(han_field.vocab.stoi)

emb_dim = 512
hidden_dim = 512
n_layers = 2

model = Model(py_vocab_size, emb_dim, hidden_dim, ch_vocab_size, n_layers).to(device)





def init_weights(m):
	for name, param in m.named_parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)

model = model.apply(init_weights)





PAD_IDX = han_field.vocab.stoi['<pad>']

criterion = nn.NLLLoss(ignore_index = PAD_IDX)


tok = ['<eos>','<unk>','<sos>','<pad>']





def compare_pre_target(output,target,show_txt=True):
	pred = torch.argmax(output,-1)
	i = 0
	for p, t in zip(pred.cpu().numpy(),target.cpu().numpy()):	

		ss = [han_field.vocab.itos[i] for i in p]
		ss = [_s for _s in ss if _s not in tok]
		s_text = ''.join(ss)

		tt = [han_field.vocab.itos[i] for i in t]
		tt = [_t for _t in tt if _t not in tok]

		t_text = ''.join(tt)

		if i ==0 and show_txt:
			print('pred:',s_text[:len(t_text)])	
			print('true:',t_text)
		i+=1
		if len(ss) !=0:
			acc = np.sum([s==t for s,t in zip(ss,tt)])/(len(tt))
		else:
			acc = 0

		return acc






def evaluate(model, iterator, criterion):

	model.eval()

	print('evaluating ....')
	val_acc = 0
	with torch.no_grad():

		for i, batch in enumerate(iterator):
			src = batch.src
			trg = batch.trg

			output = model(src)
			acc = compare_pre_target(output.detach(),trg.detach(),i % 64==0)

			#output = [batch size, trg sent len - 1, output dim]
			#trg = [batch size, trg sent len]
			#if i %32 ==0:
			   # compare_pre_target(output,trg)
			output = output.contiguous().view(-1, output.shape[-1])
			trg = trg.contiguous().view(-1)#[:,1:]

			val_acc = (val_acc*i + acc)/(i+1)
			msg = 'val acc: {:.3}'.format(val_acc)
			#output = [batch size * trg sent len - 1, output dim]
			#trg = [batch size * trg sent len - 1]

	print('done')	
	return val_acc

#val_acc = evaluate(model, valid_iterator, criterion)





n_epoch = 100
grad_clip = 1.0
SAVE_DIR = 'models'
optimizer = optim.Adam(model.parameters(),lr=3e-4)





def train():
	best_val_acc = 0.0
	for epoch in range(n_epoch):

		model.train()
		epoch_loss = 0
		epoch_acc = 0
		for i, batch in enumerate(train_loader):
			src = batch.src
			trg = batch.trg
			if trg.shape[0] ==0:
				continue
			optimizer.zero_grad()

			output = model(src)

			#if i %1024==0:
			show_txt=(i%1024==0)
			acc = compare_pre_target(output.detach(),trg.detach(),show_txt)


			#trg = [trg sent len, batch size]
			#output = [trg sent len, batch size, output dim]

			output = output[:].view(-1, output.shape[-1])
			trg = trg[:].view(-1)

			if trg.shape[0] ==0:
				continue

			#trg = [(trg sent len - 1) * batch size]
			#output = [(trg sent len - 1) * batch size, output dim]
			loss = criterion(output, trg)

			loss.backward()

			#torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

			optimizer.step()

			epoch_loss = (epoch_loss*i + loss.item())/(i+1)
			epoch_acc = (epoch_acc*i + acc)/(i+1)

			msg = 'loss:{:.5},acc:{:.5}'.format(epoch_loss,epoch_acc)

	  #  train_loss = epoch_loss
		val_acc = evaluate(model, valid_loader, criterion)


		optimizer.param_groups[0]['lr'] *= 0.9
		print('lr:',optimizer.param_groups[0]['lr'])
		if val_acc > best_val_acc:
			model_path = './models/' + 'py2han_sd_model_epoch{}val_acc{:.3}.pth'.format(epoch,val_acc)

			print('validation acc increased from {} to {},saving model to {}'.format(best_val_acc,val_acc,
				  model_path))

			best_val_acc = val_acc
			torch.save(model.state_dict(), model_path)





if __name__ == '__main__':
	train()
