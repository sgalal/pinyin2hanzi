# -*- coding: utf-8 -*-

import argparse
import ToJyutping
import re
import random

han_regex = re.compile(r'^[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f]+$')

def is_han_string(s):
	return bool(han_regex.match(s))

def preprocess(file_path):
	import io, json

	def han_to_pinyin(s):
		return ' '.join(py for _, py in ToJyutping.get_jyutping_list(s))

	with open(file_path) as f:
		for x in json.load(f):
			for line in x.splitlines():
				for s in re.split('，|。|！|？', line):
					if is_han_string(s):
						yield s, han_to_pinyin(s)

def make_pycantonese():
	import pycantonese as pc

	corpus = pc.hkcancor()
	d = {}  # Use dict instead of set to preserve insert order

	for sent in corpus.tagged_sents():
		words = []
		jyutpings = []
		for word, _, jyutping, _ in sent:
			if is_han_string(word):
				jyutping_segments = re.findall(r'[a-z]+\d', jyutping, re.UNICODE)
				if len(word) == len(jyutping_segments):
					for word_, jyutping_ in zip(word, jyutping_segments):
						words.append(word_)
						jyutpings.append(jyutping_)
		d[''.join(words), ' '.join(jyutpings)] = None

	for words, jyutpings in d:
		if words:
			yield words, jyutpings

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', default='./input/train.json', type=str, required=False, help='AI shell transcript')
	args = parser.parse_args([])

	dataset = list(preprocess(args.file)) + list(make_pycantonese())
	random.Random(42).shuffle(dataset)
	n_train = int(len(dataset) * 0.8)
	train_set = dataset[:n_train]
	dev_set = dataset[n_train:]

	with open('./data/ai_shell_train_sd.han', 'w') as f, open('./data/ai_shell_train_sd.pinyin', 'w') as g:
		for hanzi, pinyin in train_set:
			print(hanzi, file=f)
			print(pinyin, file=g)

	with open('./data/ai_shell_dev_sd.han', 'w') as f, open('./data/ai_shell_dev_sd.pinyin', 'w') as g:
		for hanzi, pinyin in dev_set:
			print(hanzi, file=f)
			print(pinyin, file=g)

	print('%d lines processed,data saved to ./data/ai_shell_' % len(dataset))

if __name__ == '__main__':
	main()
