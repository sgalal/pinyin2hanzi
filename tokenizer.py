class CharLevelTokenizer:
	'''Perform character level tokenization.'''
	TOK_UNK = 0
	TOK_SOS = 1
	TOK_EOS = 2
	TOK_PAD = 3

	def __init__(self, data_length=54):
		self.data_length = data_length
		self.d_stoi = {'<unk>': 0, '<sos>': 1, '<eos>': 2, '<pad>': 3}
		self.d_itos = ['<unk>', '<sos>', '<eos>', '<pad>']

	def build_vocab(self, texts):
		def process_char(c):
			avail_id = len(self.d_itos)
			v = self.d_stoi.setdefault(ord(c), avail_id)
			if v == avail_id:  # inserted
				self.d_itos.append(ord(c))
			return v
		def process_text(text):
			if len(text) > self.data_length - 2:
				return
			else:
				res = [self.TOK_SOS]
				for c in text:
					res.append(process_char(c))
				res.append(self.TOK_EOS)
				while len(res) < self.data_length:
					res.append(self.TOK_PAD)
				return res
		def process_texts(texts):
			for text in texts:
				res = process_text(text)
				if res:
					yield res
		return list(process_texts(texts))

	def stoi(self, texts):
		def process_char(c):
			return self.d_stoi.get(ord(c), 0)
		def process_text(text):
			if len(text) > self.data_length - 2:
				return
			else:
				res = [self.TOK_SOS]
				for c in text:
					res.append(process_char(c))
				res.append(self.TOK_EOS)
				while len(res) < self.data_length:
					res.append(self.TOK_PAD)
				return res
		def process_texts(texts):
			for text in texts:
				res = process_text(text)
				if res:
					yield res
		return list(process_texts(texts))

	def itos(self, sequences):
		return [''.join(self.to_char(item) for item in sequence) for sequence in sequences]

	def from_char(self, ch):
		return self.d_stoi.get(ch, 0)

	def to_char(self, item):
		try:
			if item < 4:
				return self.d_itos[item]
			else:
				return chr(self.d_itos[item])
		except IndexError:
			return self.d_itos[self.TOK_UNK]

	def save_vocab(self, path):
		with open(path, 'w') as f:
			for item in self.d_itos[4:]:
				print(chr(item), file=f)

	def load_vocab(self, path):
		with open(path) as f:
			for line in f:
				k = ord(line[0])
				self.d_stoi[k] = len(self.d_itos)
				self.d_itos.append(k)

	def vocab_size(self):
		return len(self.d_itos)

'''
>>> tokenizer = CharLevelTokenizer()
>>> tokenizer.build_vocab(['啊啊啊呀呀', 'aaaabbbb你好你好X'])
[[4, 4, 4, 5, 5], [6, 6, 6, 6, 7, 7, 7, 7, 8, 9, 8, 9, 10]]
>>> tokenizer.stoi(['啊啊啊呀呀', 'aaaabbbb你好你好g你好h你好X'])
[[4, 4, 4, 5, 5], [6, 6, 6, 6, 7, 7, 7, 7, 8, 9, 8, 9, 0, 8, 9, 0, 8, 9, 10]]
>>> tokenizer.itos([[4, 4, 4, 5, 5], [6, 6, 6, 6, 7, 7, 7, 7, 8, 9, 8, 9, 0, 8, 9, 0, 8, 9, 10]])
['啊啊啊呀呀', 'aaaabbbb你好你好<unk>你好<unk>你好X']
>>> tokenizer.save_vocab('vocab.txt')
'''
