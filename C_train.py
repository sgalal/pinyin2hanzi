import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import config as CONFIG
from dataset import SentenceDataset
from itos import Itos
from model import Model

# Initialize

torch.backends.cudnn.deterministic = True

train_set = SentenceDataset(train=True)
val_set = SentenceDataset(train=False)

train_loader = DataLoader(train_set, batch_size=CONFIG.BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=CONFIG.BATCH_SIZE, shuffle=False, pin_memory=True)

total_batch = len(train_loader)

itos_x = Itos('data/vocab_x.txt')
itos_y = Itos('data/vocab_y.txt')

x_vocab_size = itos_x.vocab_size()
y_vocab_size = itos_y.vocab_size()

model = Model(x_vocab_size, CONFIG.EMB_SIZE, CONFIG.HIDDEN_SIZE, y_vocab_size, CONFIG.NUM_LAYERS, CONFIG.DROPOUT)

trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_loader, val_loader)
