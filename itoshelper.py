import config as CONFIG

class ItosHelper:
	TOK_UNK = 0
	TOK_SOS = 1
	TOK_EOS = 2
	TOK_PAD = 3

	def __init__(self, path):
		with open(path) as f:
			self.data = {ord(line[0]): i for i, line in enumerate(f, start=4)}

	def stoi(self, sequence):
		assert len(sequence) <= CONFIG.PAD_TO - 2, 'Sequence is too long'
		res = [self.TOK_SOS] + [self.data.get(ord(c), self.TOK_UNK) for c in sequence]
		res += [self.TOK_PAD] * (CONFIG.PAD_TO - len(res) - 1)
		res += [self.TOK_EOS]
		return res

	def vocab_size(self):
		return 4 + len(self.data)
