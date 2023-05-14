# Librispeech Dataset

import os
import random
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
import torchaudio

class LibriDataset(Dataset):
    """Librispeech Waveform Dataset for Distillation"""

    def __init__(
        self,
        batch_size,
        file_path='/workspace/s3prl/s3prl/data/len_for_bucket/',
        sets=['train-clean-100', 'train-clean-360', 'train-other-500'],
        libri_root='/workspace/LibriSpeech/'
    ):
        super().__init__()

        self.libri_root = libri_root

        # Read file
        self.root = file_path
        tables = [pd.read_csv(os.path.join(file_path, s + ".csv")) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(
            by=["length"], ascending=False
        )
        print("[Dataset] - Training data from these sets:", str(sets))

        X = self.table["file_path"].tolist()
        X_lens = self.table["length"].tolist()
        self.num_samples = len(X)
        print("[Dataset] - Number of individual training instances:", self.num_samples)

        # Use bucketing to allow different batch size at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            batch_x.append(x)
            batch_len.append(x_len)

            # Fill in batch_x until batch is full
            if len(batch_x) == batch_size:
                self.X.append(batch_x)
                batch_x, batch_len = [], []

        # Gather the last batch
        if len(batch_x) > 1:
            self.X.append(batch_x)

    def collate_fn(self, items):
        items = items[0]
        return items

    def _load_feat(self, feat_path):
        wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))
        return wav.squeeze()

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wave_orig = [self._load_feat(x_file) for x_file in self.X[index]]

        wav_lengths = torch.LongTensor([len(wav) for wav in wave_orig])
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wave_orig, batch_first=True)

        return {'x': padded_wav, 'padding_mask': wav_padding_mask}

    def __len__(self):
        return len(self.X)
