import torch
from torch import nn

from Decoder import Decoder
from Encoder import Encoder


## Only Encoders architecture for audio classification
class Transformer(nn.Module):

    def __init__(
        self,
        src_pad_idx,
        enc_voc_size,
        d_model,
        n_head,
        max_len,
        ffn_hidden,
        n_layers,
        drop_prob,
        device,
    ):
        super().__init__()
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.encoder = Encoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device,
        )
        self.fc_output = 128
        self.fc = nn.Linear(d_model, self.fc_output)
        self.relu = nn.ReLU()
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.classification = nn.Linear(self.fc_output, 1)

    def forward(self, src):
        # Convert src to float due to MLFlow json conversion error
        src = src.float().to(self.device)
        src_mask = self.make_src_mask(src)
        output = self.encoder(src, src_mask)

        # Add layer for binary classification
        output = self.relu(self.fc(output))
        output = self.avg_pooling(output.transpose(1, 2)).squeeze(-1)
        output = self.classification(output)

        return output.squeeze(-1)

    def make_src_mask(self, src):
        src_mask = (
            (src != self.src_pad_idx)
            .all(dim=-1, keepdim=True)
            .unsqueeze(1)
            .unsqueeze(2)
        )
        # (N, 1, 1, src_len)
        src_mask = src_mask.expand(-1, -1, src.size(1), -1, -1).squeeze(-1)
        return src_mask


# AdamW
# https://github.com/hyunwoongko/transformer/blob/master/util/data_loader.py
