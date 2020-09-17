import torch

import config as CONFIG

class UniqueIDGen:
	def __init__(self, start=0):
		self.i = start
		self.d = {}
		self.l = []

	def get_or_create(self, e):
		try:
			return self.d[e]
		except KeyError:
			i = self.i
			self.d[e] = i
			self.l.append(e)
			self.i += 1
			return i

	def get(self, e, default=None):
		return self.d.get(e, default)

class CharLevelTokenizer:
	'''Perform character level tokenization.'''
	TOK_UNK = 0
	TOK_SOS = 1
	TOK_EOS = 2
	TOK_PAD = 3

	def __init__(self):
		self.uig = UniqueIDGen(start=4)

	def build_vocab(self, texts):
		xss = []
		for line in texts:
			xs = [self.TOK_SOS]
			for c in line:
				cp = ord(c)
				i = self.uig.get_or_create(cp)
				xs.append(i)
			xs.append(self.TOK_EOS)
			xs.extend((CONFIG.PAD_TO - len(xs)) * [self.TOK_PAD])
			xss.append(xs)
		return xss

	def get_vocab(self, texts):
		xss = []
		for line in texts:
			xs = [self.TOK_SOS]
			for c in line:
				cp = ord(c)
				i = self.uig.get(cp, self.TOK_UNK)
				xs.append(i)
			xs.append(self.TOK_EOS)
			xs.extend((CONFIG.PAD_TO - len(xs)) * [self.TOK_PAD])
			xss.append(xs)
		return xss

	def get_vocab_list(self):
		return self.uig.l

def build_vocab_y():
	clt = CharLevelTokenizer()
	with open('data/train_y.txt') as f:
		ys = (line.rstrip('\n') for line in f)
		tokens_train_y = clt.build_vocab(ys)
	torch.save(torch.tensor(tokens_train_y), 'data/tokens_train_y.pth')

	with open('data/test_y.txt') as f:
		ys = (line.rstrip('\n') for line in f)
		tokens_test_y = clt.get_vocab(ys)
	torch.save(torch.tensor(tokens_test_y), 'data/tokens_test_y.pth')

	with open('data/vocab_y.txt', 'w') as f:
		for line in clt.get_vocab_list():
			print(chr(line), file=f)

def build_vocab_x():
	clt = CharLevelTokenizer()
	with open('data/train_x.txt') as f:
		xs = (line.rstrip('\n') for line in f)
		tokens_train_x = clt.build_vocab(xs)
	torch.save(torch.tensor(tokens_train_x), 'data/tokens_train_x.pth')

	with open('data/test_x.txt') as f:
		xs = (line.rstrip('\n') for line in f)
		tokens_test_x = clt.get_vocab(xs)
	torch.save(torch.tensor(tokens_test_x), 'data/tokens_test_x.pth')

	with open('data/vocab_x.txt', 'w') as f:
		for line in clt.get_vocab_list():
			print(chr(line), file=f)

build_vocab_y()
build_vocab_x()
