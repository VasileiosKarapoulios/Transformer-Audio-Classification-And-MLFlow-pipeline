from torch import nn
import torch

from embedding.positional_encoding import PositionalEncoding
from embedding.token_embeddings import TokenEmbedding

CUDA_LAUNCH_BLOCKING = 1


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.conv1d = nn.Conv1d(
            in_channels=vocab_size,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # Assuming MFCC features of size 40
        self.relu = nn.ReLU()
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # Token embedding here is conv1d since MFCC is 2d
        tok_emb = self.relu(self.conv1d(x.transpose(1, 2))).transpose(1, 2)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
