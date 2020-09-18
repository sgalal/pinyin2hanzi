from itertools import repeat
import torch

import config as CONFIG
from uniqueid import UniqueID

class CharLevelTokenizer:
	'''Perform character level tokenization.'''
	TOK_UNK = 0
	TOK_SOS = 1
	TOK_EOS = 2
	TOK_PAD = 3

	def __init__(self):
		self.uniqueid = UniqueID(start=4)

	def build_vocab(self, texts):
		def inner(s):
			yield self.TOK_SOS
			yield from (self.uniqueid.get_or_create(ord(c)) for c in s)
			yield self.TOK_EOS
			yield from repeat(self.TOK_PAD, CONFIG.PAD_TO - 2 - len(s))
		return [list(inner(line)) for line in texts]

	def get_vocab(self, texts):
		def inner(s):
			yield self.TOK_SOS
			yield from (self.uniqueid.get(ord(c), self.TOK_UNK) for c in s)
			yield self.TOK_EOS
			yield from repeat(self.TOK_PAD, CONFIG.PAD_TO - 2 - len(s))
		return [list(inner(line)) for line in texts]

	def save_vocab_list(self, f):
		for line in self.uniqueid.i2s:
			print(chr(line), file=f)

def build_vocab(in_a, in_b, out_a, out_b, vocab_file):
	tokenizer = CharLevelTokenizer()

	with open(in_a) as f:
		tokens_train = tokenizer.build_vocab(line.rstrip('\n') for line in f)
		torch.save(torch.tensor(tokens_train), out_a)

	with open(in_b) as f:
		tokens_test = tokenizer.get_vocab(line.rstrip('\n') for line in f)
		torch.save(torch.tensor(tokens_test), out_b)

	with open(vocab_file, 'w') as f:
		tokenizer.save_vocab_list(f)

build_vocab('data/train_x.txt', 'data/test_x.txt', 'data/tokens_train_x.pth', 'data/tokens_test_x.pth', 'data/vocab_x.txt')
build_vocab('data/train_y.txt', 'data/test_y.txt', 'data/tokens_train_y.pth', 'data/tokens_test_y.pth', 'data/vocab_y.txt')
