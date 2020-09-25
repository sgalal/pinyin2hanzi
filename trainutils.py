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
	print('SI:', itos_x(sample_x))  # sample input
	print('EO:', itos_y(sample_y))  # expected output
	print('MO:', itos_y(sample_y_hat))  # model output

# Training process

def train(train_loader, model, criterion, optimizer, total_batch, current_epoch, itos_x, itos_y):
	total_loss = 0
	for current_batch, (x, y) in enumerate(train_loader):
		y_hat = model(x)
		loss = criterion(y_hat, y)
		current_loss = loss.item()
		total_loss += current_loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if current_batch % 10 == 0:
			print('Epoch', current_epoch, 'batch %d/%d:' % (current_batch, total_batch), 'loss', current_loss)
			show_sample_result(x, y, y_hat, itos_x, itos_y)
	print('Epoch', current_epoch, 'train total loss:', total_loss)

def test(test_loader, model, criterion, current_epoch, itos_x, itos_y):
	total_loss = 0
	with torch.no_grad():
		for x, y in test_loader:
			y_hat = model(x)
			loss = criterion(y_hat, y)
			total_loss += loss.item()
		print('Epoch', current_epoch, 'test loss:', total_loss)
		show_sample_result(x, y, y_hat, itos_x, itos_y)

def save(current_epoch, model, optimizer):
	state = {
		'epoch': current_epoch,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'rng_state': torch.get_rng_state(),
	}
	torch.save(state, CONFIG.MODEL_PATH)
	print('Saved epoch', current_epoch)
