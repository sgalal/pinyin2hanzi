import ToJyutping

def sanitize_py(s):
	for c in '123456 ':
		s = s.replace(c, '')
	return s

with open('data/corpus.txt') as f, \
	open('data/train_x.txt', 'w') as train_file_x, \
	open('data/train_y.txt', 'w') as train_file_y, \
	open('data/test_x.txt', 'w') as test_file_x, \
	open('data/test_y.txt', 'w') as test_file_y:
	for line in f:
		a_s = []
		b_s = []
		for k, v in ToJyutping.get_jyutping_list(line.rstrip('\n')):
			if v is None:
				a = k.lower()
				b = k
			else:
				a = sanitize_py(v)
				b = '-' * (len(a) - 1) + k
			a_s.append(a)
			b_s.append(b)
		a_s = ''.join(a_s)
		b_s = ''.join(b_s)
		print(a_s, file=train_file_x)
		print(b_s, file=train_file_y)
		print(a_s, file=test_file_x)
		print(b_s, file=test_file_y)
