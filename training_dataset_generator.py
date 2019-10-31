import numpy as np

class Training_data:
    def __init__(self, data_length):
        self.seq_len = data_length
        self.input_seq = np.zeros((self.seq_len, 3))
        self.output_seq = np.zeros((self.seq_len,))

    def



