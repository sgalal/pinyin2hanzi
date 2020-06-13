# -*- coding: utf-8 -*-

import argparse
import random
import re
import ToJyutping

from itertools import chain

from opencc import OpenCC
trad2hk = OpenCC('t2hk')  # trad2hk.convert('臺灣') -> '台灣'

# https://ayaka.shn.hk/hanregex/
han_regex = re.compile(r'^[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f]+$')

def is_han_string(s):
	return bool(han_regex.match(s))

def han_to_pinyin(s):
	return ' '.join(py[:-1] for _, py in ToJyutping.get_jyutping_list(s))  # 去聲調

def make_news2_data():
	with open('input/data_aishell_transcript.txt') as f:
		for s in f:
			s = s.rstrip()
			if (len(s) - s.count(' ')) <= 18:  # Restrict max length to 18
				try:
					yield trad2hk.convert(s).replace(' ', ''), ' '.join(han_to_pinyin(x) for x in s.split(' '))
				except:
					pass

def make_news_data():
	with open('input/data_yuewiki.txt') as f1, open('input/data_rthk_1.txt') as f2:
		for s in chain(f1, f2):
			s = s.rstrip()
			if len(s) <= 18:  # Restrict max length to 18
				try:
					yield s, han_to_pinyin(s)
				except:
					pass

def make_pycantonese():
	import pycantonese as pc
	corpus = pc.hkcancor()

	for sent in corpus.tagged_sents():
		words = []
		jyutpings = []
		for word, _, jyutping, _ in sent:
			if word in (',', '?', '.'):  # 從 ,?. 分隔整個句子
				word_ = ''.join(words)
				if 1 < len(word_) < 18:
					yield word_, ' '.join(jyutpings)

				words = []
				jyutpings = []
			if is_han_string(word):
				jyutping_segments = re.findall(r'[a-z]+\d', jyutping, re.UNICODE)
				if len(word) == len(jyutping_segments):
					for word_, jyutping_ in zip(word, jyutping_segments):
						words.append(word_)
						jyutpings.append(jyutping_[:-1])  # 去聲調
		word_ = ''.join(words)
		if 1 < len(word_) < 18:
			yield word_, ' '.join(jyutpings)

def postprocess_hzpy(hanzi, pinyin):
	hanzis = list(hanzi)
	pinyins = pinyin.split(' ')
	assert len(hanzis) == len(pinyins)

	hanzis_ = []
	pinyins_ = []
	for h, ps in zip(hanzis, pinyins):
		assert len(ps) > 0
		for _ in range(len(ps) - len(h)):
			hanzis_.append('x')
		hanzis_.append(h)
		pinyins_.append(ps)

	return ''.join(hanzis_), ''.join(pinyins_)

def random_binary_distr(n):
	num_of_zero = int(n * 0.8)
	num_of_one = n - num_of_zero
	arr = [0] * num_of_zero + [1] * num_of_one
	random.shuffle(arr)
	return arr

def postprocess_hzpy_mask(hanzi, pinyin):
	hanzis = list(hanzi)
	pinyins = pinyin.split(' ')
	assert len(hanzis) == len(pinyins)

	mask_pos = random_binary_distr(len(hanzis))

	hanzis_ = []
	pinyins_ = []
	for h, ps, should_mask in zip(hanzis, pinyins, mask_pos):
		assert len(ps) > 0
		if should_mask:
			hanzis_.append(h)
			pinyins_.append(ps[0])
		else:
			for _ in range(len(ps) - len(h)):
				hanzis_.append('x')
			hanzis_.append(h)
			pinyins_.append(ps)

	return ''.join(hanzis_), ''.join(pinyins_)

def main():
	dataset = list(make_news_data()) + list(make_pycantonese()) + list(make_news2_data())
	random.Random(42).shuffle(dataset)
	n_train = int(len(dataset) * 0.8)
	train_set = dataset[:n_train]
	dev_set = dataset[n_train:]

	with open('./data/ai_shell_train_sd.han', 'w') as f, open('./data/ai_shell_train_sd.pinyin', 'w') as g:
		for hanzi, pinyin in train_set:
			hanzi, pinyin = postprocess_hzpy(hanzi, pinyin)
			print(hanzi, file=f)
			print(pinyin, file=g)
		for hanzi, pinyin in train_set:
			hanzi, pinyin = postprocess_hzpy_mask(hanzi, pinyin)
			print(hanzi, file=f)
			print(pinyin, file=g)

	with open('./data/ai_shell_dev_sd.han', 'w') as f, open('./data/ai_shell_dev_sd.pinyin', 'w') as g:
		for hanzi, pinyin in dev_set:
			hanzi, pinyin = postprocess_hzpy(hanzi, pinyin)
			print(hanzi, file=f)
			print(pinyin, file=g)
		for hanzi, pinyin in dev_set:
			hanzi, pinyin = postprocess_hzpy_mask(hanzi, pinyin)
			print(hanzi, file=f)
			print(pinyin, file=g)

	print('%d lines processed,data saved to ./data/ai_shell_' % len(dataset))

if __name__ == '__main__':
	main()
