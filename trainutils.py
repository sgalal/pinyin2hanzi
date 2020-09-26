import logging
import torch

import config as CONFIG

# Utilities

# Use the rand function in torch package so that we can resume the rand state
def randrange(n: int):
	return int(torch.rand(1).item() * n)

def show_sample_result(x, y, y_hat, itos_x, itos_y):
	rand_idx = randrange(y.shape[0])
	sample_x = x[rand_idx].tolist()
	sample_y = y[rand_idx].tolist()
	sample_y_hat = y_hat[rand_idx].argmax(0).tolist()
	logging.info('SI: %s', itos_x(sample_x))  # sample input
	logging.info('EO: %s', itos_y(sample_y))  # expected output
	logging.info('MO: %s', itos_y(sample_y_hat))  # model output
