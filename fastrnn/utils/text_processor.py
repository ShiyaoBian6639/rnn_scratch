import numpy as np


class TextProcess:
    def __init__(self, data: str):
        self.data = data
        self.char = list(set(data))
        self.data_size = len(self.data)
        self.vocab_size = len(self.char)
        self.char_to_idx = self.get_char_to_idx()
        self.idx_to_char = self.get_idx_to_char()
        self.data_token = self.data_to_idx()
        self.data_arr = self.data_to_one_hot_encoding()

    def get_char_to_idx(self) -> dict:
        return {ch: i for i, ch in enumerate(self.char)}

    def get_idx_to_char(self) -> dict:
        return {i: ch for i, ch in enumerate(self.char)}

    def data_to_idx(self) -> list:
        res = [self.char_to_idx[ch] for ch in self.data]
        return res

    def data_to_one_hot_encoding(self) -> np.ndarray:
        input_arr = np.zeros((self.data_size, self.vocab_size))
        for i in range(self.data_size):
            input_arr[i, self.data_token[i]] = 1
        return input_arr

