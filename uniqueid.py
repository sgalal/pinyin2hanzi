class UniqueID:
	def __init__(self, start=0):
		self.start = start
		self.s2i = {}
		self.i2s = []

	def get_or_create(self, s):
		'''
		>>> uniqueid = UniqueID(start=2)
		>>> uniqueid.get_or_create('a')
		2
		>>> uniqueid.get_or_create('b')
		3
		>>> uniqueid.get_or_create('a')
		2
		'''
		try:
			return self.s2i[s]
		except KeyError:
			i = self.start + len(self.i2s)
			self.s2i[s] = i
			self.i2s.append(s)
			return i

	def get(self, s, default=None):
		return self.s2i.get(s, default)
