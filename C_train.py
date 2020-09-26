import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

import config as CONFIG
from dataset import SentenceDataset
from itos import Itos
from model import Model

# Initialize

torch.backends.cudnn.deterministic = True

dataset = SentenceDataset()

data_len = len(dataset)
train_len = int(data_len * 0.7)
val_len = int(data_len * 0.2)
test_len = data_len - train_len - val_len

train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=CONFIG.batch_size, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=CONFIG.batch_size, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=CONFIG.batch_size, pin_memory=True)

itos_x = Itos('data/vocab_x.txt')
itos_y = Itos('data/vocab_y.txt')

x_vocab_size = itos_x.vocab_size()
y_vocab_size = itos_y.vocab_size()

model = Model(x_vocab_size, CONFIG.emb_size, CONFIG.hidden_size, y_vocab_size, CONFIG.num_layers, CONFIG.dropout)

trainer = pl.Trainer(gpus=1, fast_dev_run=True)
trainer.fit(model, train_loader, val_loader)
trainer.test(test_dataloaders=test_loader)
