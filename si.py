import os

here = os.path.dirname(os.path.abspath(__file__))

class SIHelper:
    def __init__(self, path):
        with open(os.path.join(here, path)) as f:
            self.data = ['<unk>', '<sos>', '<eos>', '<pad>'] + [line[0] for line in f]

    def get_char(self, i):
        try:
            return self.data[i]
        except IndexError:
            return self.data[0]

    def trim_tokens(self, lst):
        while lst and lst[-1] == 3:
            lst.pop()
        if lst and lst[-1] == 2:
            lst.pop()
        if lst and lst[0] == 1:
            lst.pop(0)

    def itos(self, sequence):
        self.trim_tokens(sequence)
        return ''.join(self.get_char(i) for i in sequence)

    def __len__(self):
        return len(self.data)
