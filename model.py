import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import config as CONFIG

class Model(pl.LightningModule):
	def __init__(self, input_size, emb_size, hidden_size, output_dim, num_layers, dropout):
		super().__init__()
		self.embedding = nn.Embedding(input_size, emb_size)
		self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
		self.hidden2out = nn.Linear(hidden_size * 2, output_dim)

	def forward(self, x):
		x = self.embedding(x)
		x, (hidden, cell) = self.rnn(x)
		x = self.hidden2out(x)
		x = F.log_softmax(x, dim=-1)
		x = x.permute(0, 2, 1)
		return x

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=CONFIG.learning_rate)
		return optimizer

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.nll_loss(y_hat, y)
		result = pl.TrainResult(loss)
		result.log('train_loss', loss, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.cross_entropy(y_hat, y)
		result = pl.EvalResult(checkpoint_on=loss)
		result.log('val_loss', loss, prog_bar=True)
		return result

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.cross_entropy(y_hat, y)
		result = pl.EvalResult()
		result.log('test_loss', loss)
		return result
