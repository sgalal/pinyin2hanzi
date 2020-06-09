# -*- coding: utf-8 -*-

from inference_sd import convert

if __name__ == "__main__":
	while True:
		try:
			print(convert(input('> ')))
		except EOFError:
			break
