import torch
import torch.nn as nn
import torchaudio
import os


class MFCC(nn.Module):

    def __init__(
        self,
        sample_rate,
        n_fft=512,
        hop_length=256,
        n_mels=40,
    ):
        super(MFCC, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={"n_fft": 512, "hop_length": 256, "n_mels": 40},
        )

    def forward(self, x):
        return torch.Tensor(self.mfcc(x))


def get_featurizer(sample_rate):
    return MFCC(sample_rate=sample_rate)


class RandomCut(nn.Module):
    """Augmentation technique that randomly cuts start or end of audio"""

    def __init__(self, max_cut=10):
        super(RandomCut, self).__init__()
        self.max_cut = max_cut

    def forward(self, x, max_len):
        """Randomly cuts from start or end of batch"""
        side = torch.randint(0, 1, (1,))
        cut = torch.randint(1, self.max_cut, (1,))
        if side == 0:
            x = x[:, :-cut, :]
        elif side == 1:
            x = x[:, cut:, :]

        # Pad or truncate the sequence length to ensure consistency
        max_seq_len = max_len
        if x.size(1) > max_seq_len:
            x = x[:, :max_seq_len, :]  # Truncate if longer than max_seq_len
        elif x.size(1) < max_seq_len:
            pad = torch.zeros(
                x.size(0), max_seq_len - x.size(1), x.size(2)
            )  # Pad with zeros if shorter than max_seq_len
            x = torch.cat([x, pad], dim=1)

        return x


class SpecAugment(nn.Module):
    """Augmentation technique to add masking on the time or frequency domain"""

    def __init__(self, rate, policy=3, freq_mask=2, time_mask=4):
        super(SpecAugment, self).__init__()

        self.rate = rate

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
        )

        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
        )

        policies = {1: self.policy1, 2: self.policy2, 3: self.policy3}
        self._forward = policies[policy]

    def forward(self, x):
        return self._forward(x)

    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            x = self.specaug(x)
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            x = self.specaug2(x)
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)


class DataClass(torch.utils.data.Dataset):
    """Load and process wakeword data"""

    def __init__(self, data_path, sample_rate=8000, max_len=862, valid=False):
        self.sr = sample_rate
        self.data = {}

        if valid:
            self.audio_transform = get_featurizer(sample_rate)
        else:
            self.audio_transform = nn.Sequential(
                get_featurizer(sample_rate), SpecAugment(rate=0.5)
            )

        # Scan all directory and collect .wav files
        # Walk through the directory tree
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".wav") or file.endswith(".mp3"):
                    # Get the folder the file belongs to
                    relative_path = os.path.relpath(root, data_path)
                    parts = relative_path.split(os.sep)

                    # Assume that the folder we are interested in is always the immediate child of the base path
                    if len(parts) >= 1 and parts[0] in ["0", "1"]:
                        folder = parts[0]
                    elif len(parts) >= 2 and parts[1] in ["0", "1"]:
                        folder = parts[1]
                    else:
                        continue

                    # Add the file to the dictionary if the limit for that folder is not yet reached
                    full_path = os.path.join(root, file)
                    self.data[full_path] = int(folder)

        self.file_paths = list(self.data.keys())  # Extract keys once and store them
        self.labels = list(self.data.values())  # Extract values once and store them
        print("Loaded Dataset")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        if sr > self.sr:
            waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)

        # Convert stereo audio to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        mfcc = self.audio_transform(waveform)

        label = self.labels[idx]

        return mfcc, label

    def __len__(self):
        return len(self.data)


rand_cut = RandomCut(max_cut=10)


def collate_fn(data, max_len, valid):
    """Batch and pad wakeword data"""
    mfccs = []
    labels = []
    for d in data:
        mfcc, label = d
        mfccs.append(mfcc.squeeze(0).transpose(0, 1))
        labels.append(label)

    # Pad or truncate mfccs to ensure all tensors have sequence length of max_len
    max_seq_len = max_len
    mfccs_padded = []
    for mfcc in mfccs:
        if mfcc.size(0) > max_seq_len:
            mfccs_padded.append(
                mfcc[:max_seq_len, :]
            )  # Truncate if longer than max_len
        elif mfcc.size(0) < max_seq_len:
            pad = torch.zeros(
                max_seq_len - mfcc.size(0), mfcc.size(1)
            )  # Pad with zeros if shorter than max_len
            mfccs_padded.append(torch.cat([mfcc, pad], dim=0))
        else:
            mfccs_padded.append(mfcc)  # No padding or truncation needed

    # Stack padded MFCCs into a single tensor
    mfccs_padded = torch.stack(mfccs_padded)
    if not valid:
        mfccs_padded = rand_cut(mfccs_padded, max_len)
    labels = torch.Tensor(labels)
    return mfccs_padded, labels
