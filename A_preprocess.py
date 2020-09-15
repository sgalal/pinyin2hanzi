import random
import ToJyutping

import config as CONFIG

simplify_rate = 0.2

random.seed(42)

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

lines_total = get_line_count('data/corpus.txt')
lines_train_set = int(lines_total * 0.8)

def process_jyutping_list(xs):
	res = []

	# Full romanisation
	xxs = []
	yys = []
	for i, (y, x) in enumerate(xs):
		if x is None:
			a = y.lower()
			b = y
		else:
			a = remove_tones_and_spaces(x)
			b = '-' * (len(a) - 1) + y
		xxs.append(a)
		yys.append(b)
	xxs = ''.join(xxs)
	yys = ''.join(yys)

	if len(yys) > CONFIG.PAD_TO - 2:
		return res

	res.append((xxs, yys))

	# Simplified romanisation
	should_simplify = random.sample(range(len(xs)), int(len(xs) * simplify_rate))

	xxs = []
	yys = []
	for i, (y, x) in enumerate(xs):
		if x is None:
			a = y.lower()
			b = y
		else:
			if i in should_simplify:
				a = x[0]
				b = y
			else:
				a = remove_tones_and_spaces(x)
				b = '-' * (len(a) - 1) + y
		xxs.append(a)
		yys.append(b)
	xxs = ''.join(xxs)
	yys = ''.join(yys)

	res.append((xxs, yys))

	return res

with open('data/corpus.txt') as f, \
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

		for xxs, yys in process_jyutping_list(ToJyutping.get_jyutping_list(line.rstrip('\n'))):
			print(xxs, file=file_x)
			print(yys, file=file_y)
