import config as CONFIG

class Stoi:
	TOK_UNK = 0
	TOK_SOS = 1
	TOK_EOS = 2
	TOK_PAD = 3

	def __init__(self, path):
		with open(path) as f:
			self.data = {ord(line[0]): i for i, line in enumerate(f, start=4)}

	def __call__(self, s):
		assert len(s) <= CONFIG.PAD_TO - 2, 'Sequence is too long'
		def inner():
			yield self.TOK_SOS
			yield from (self.data.get(ord(c), self.TOK_UNK) for c in s)
			yield from (self.TOK_PAD for _ in range(CONFIG.PAD_TO - 2 - len(s)))
			yield self.TOK_EOS
		return list(inner())

	def vocab_size(self):
		return 4 + len(self.data)
