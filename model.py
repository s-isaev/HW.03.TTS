import torch
from torch import Tensor, nn
import math

class PositionalEncoding(nn.Module):
    def __init__(
        self, emb_size: int, dropout: float, maxlen: int = 5000):

        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size) -> None:
        super(TransformerBlock, self).__init__()

        self.net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2), 
            num_layers=2
        )

    def forward(self, x):
        return self.net(x)


class LengthRegulator(nn.Module):

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, x, mels_alignations, mels_gt_num, token_nums):
        mels_alignations = (mels_alignations.T/mels_alignations.sum(axis=1)*mels_gt_num).T
        # print("Normalised")
        # print(mels_alignations)
        # print("Mels gt num:", mels_gt_num)
        x_new = torch.zeros((x.shape[0], int(mels_gt_num.max().item()), x.shape[2])).to('cuda:0')
        for examle in range(x.shape[0]):
            cur_mel = 0
            cur_tgt = 0
            for i in range(x.shape[1]):
                cur_tgt += mels_alignations[examle][i]
                cur_tgt = min(cur_tgt, mels_gt_num[examle])
                while cur_mel < cur_tgt:
                    x_new[examle][cur_mel] = x[examle][i]
                    cur_mel += 1
        # print("x_new:")
        # print(x_new)
        # print("x_new shape:", x_new.shape)
        return x_new



class FastSpeech(nn.Module):

    def __init__(self, vocab_size, hidden_size, n_mels):
        super(FastSpeech, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=0
        )

        self.fir_position_enc = PositionalEncoding(
            emb_size=hidden_size, dropout=0.0
        )

        self.encoder_fft = TransformerBlock(hidden_size=hidden_size)

        self.length_regulator = LengthRegulator()

        self.sec_position_enc = PositionalEncoding(
            emb_size=hidden_size, dropout=0.0
        )

        self.decoder_fft = TransformerBlock(hidden_size=hidden_size)

        self.linear_layer = nn.Linear(hidden_size, n_mels)

    def forward(self, tokens, mels_alignations, mels_gt_num, token_nums):
        x = self.embedding(tokens)
        x = self.fir_position_enc(x)
        x = self.encoder_fft(x)
        x = self.length_regulator(x, mels_alignations, mels_gt_num, token_nums)
        x = self.sec_position_enc(x)
        x = self.decoder_fft(x)
        x = self.linear_layer(x)
        return x
