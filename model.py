import torch
import torch.nn as nn

class Model(nn.Module):
	def __init__(self, input_size, emb_size, hidden_size, output_dim, num_layers, dropout):
		super().__init__()
		self.embedding = nn.Embedding(input_size, emb_size)
		self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
		self.hidden2out = nn.Linear(hidden_size * 2, output_dim)
		self.softmax = nn.LogSoftmax(dim=-1)

	def forward(self, src):
		output = self.embedding(src)
		output, (hidden, cell) = self.rnn(output)
		output = self.hidden2out(output)
		output = self.softmax(output)
		output = output.permute(0, 2, 1)
		return output

	def init_weights(self):
		def inner(model: nn.Module):
			for param in model.parameters():
				nn.init.uniform_(param.data, -0.08, 0.08)
		self.apply(inner)
