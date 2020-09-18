class Itos:
	TOK_UNK = 0
	TOK_SOS = 1
	TOK_EOS = 2
	TOK_PAD = 3

	def __init__(self, path):
		from itertools import chain
		with open(path) as f:
			self.data = list(chain(('<unk>', '<sos>', '<eos>', '<pad>'), (line[0] for line in f)))

	def __call__(self, lst):
		# Trim tokens
		while lst and lst[-1] == self.TOK_PAD:
			lst.pop()
		if lst and lst[-1] == self.TOK_EOS:
			lst.pop()
		if lst and lst[0] == self.TOK_SOS:
			lst.pop(0)

		def inner(i):
			try:
				return self.data[i]
			except IndexError:
				return self.data[0]

		return ''.join(inner(i) for i in lst)

	def vocab_size(self):
		return len(self.data)
