import ToJyutping

def has_number(s):
	'''
	Returns true if a string contains at least one number.
	Otherwise returns false.
	'''
	return any(c.isdigit() for c in s)

def get_line_count(path):
	with open(path) as f:
		return sum(1 for _ in f)

def remove_tones_and_spaces(s):
	'''
	>>> remove_tones_and_spaces('cin1 ngaa5')
	'cinngaa'
	'''
	for c in '123456 ':
		s = s.replace(c, '')
	return s

lines_total = get_line_count('data/corpus_shuffled.txt')
lines_train_set = int(lines_total * 0.8)

with open('data/corpus_shuffled.txt') as f, \
open('data/train_x.txt', 'w') as train_file_x, \
open('data/train_y.txt', 'w') as train_file_y, \
open('data/test_x.txt', 'w') as test_file_x, \
open('data/test_y.txt', 'w') as test_file_y:
	for i, line in enumerate(f):
		if has_number(line):
			continue

		if i < lines_train_set:
			file_x = train_file_x
			file_y = train_file_y
		else:
			file_x = test_file_x
			file_y = test_file_y

		a_s = []
		b_s = []
		for k, v in ToJyutping.get_jyutping_list(line.rstrip('\n')):
			if v is None:
				a = k.lower()
				b = k
			else:
				a = remove_tones_and_spaces(v)
				b = '-' * (len(a) - 1) + k
			a_s.append(a)
			b_s.append(b)
		a_s = ''.join(a_s)
		b_s = ''.join(b_s)

		print(a_s, file=file_x)
		print(b_s, file=file_y)
