# -*- coding: utf-8 -*-

import argparse
import ToJyutping
import re
import numpy as np

np.random.seed(666)

han_regex = re.compile(r'^[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f]+$')

def is_han_string(s):
	return bool(han_regex.match(s))

def preprocess(file_path):
	import io, json

	with open(file_path) as f:
		for x in json.load(f):
			for line in x.splitlines():
				for s in re.split('，|。|！|？', line):
					if is_han_string(s):
						yield s

def han_to_pinyin(s):
	return ' '.join(py for _, py in ToJyutping.get_jyutping_list(s))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', default='./train.json',
						type=str, required=False, help='AI shell transcript')
	args = parser.parse_args([])

	lines = list(preprocess(args.file))

	#idx = np.random.permutation(n)
	#lines = [''.join(lines[i]) for i in idx]

	n_train = int(len(lines) * 0.8)

	with open('./data/ai_shell_train_sd.han', 'w') as f:
		for line in lines[:n_train]:
			print(line, file=f)

	with open('./data/ai_shell_train_sd.pinyin', 'w') as f:
		for line in lines[:n_train]:
			print(han_to_pinyin(line), file=f)

	with open('./data/ai_shell_dev_sd.han', 'w') as f:
		for line in lines[n_train:]:
			print(line, file=f)

	with open('./data/ai_shell_dev_sd.pinyin', 'w') as f:
		for line in lines[n_train:]:
			print(han_to_pinyin(line), file=f)

	print('%d lines processed,data saved to ./data/ai_shell_' % len(lines))

if __name__ == '__main__':
	main()
