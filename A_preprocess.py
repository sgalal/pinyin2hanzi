import random
import ToJyutping

import config as CONFIG

random.seed(42)

def has_number(s):
	'''
	Returns true if a string contains at least one number.
	Otherwise returns false.
	'''
	return any(c.isdigit() for c in s)

def remove_tones_and_spaces(s):
	'''
	>>> remove_tones_and_spaces('cin1 ngaa5')
	'cinngaa'
	'''
	for c in '123456 ':
		s = s.replace(c, '')
	return s

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

	if len(yys) > CONFIG.pad_to - 2:
		return res

	res.append((xxs, yys))

	return res

with open('data/corpus.txt') as f, \
open('data/x.txt', 'w') as file_x, \
open('data/y.txt', 'w') as file_y:
	for i, line in enumerate(f):
		if has_number(line):
			continue

		for xxs, yys in process_jyutping_list(ToJyutping.get_jyutping_list(line.rstrip('\n'))):
			print(xxs, file=file_x)
			print(yys, file=file_y)
