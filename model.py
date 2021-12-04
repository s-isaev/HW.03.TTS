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

    def __init__(self, hidden_size):
        super(LengthRegulator, self).__init__()

        
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=11,
            padding='same'
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=11,
            padding='same'
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, token_nums, mels_alignations=None, mels_gt_num=None):
        m = self.conv1(x.transpose(1,2)).transpose(1,2)
        m = self.relu1(self.ln1(m))
        m = self.conv2(m.transpose(1,2)).transpose(1,2)
        m = self.relu2(self.ln2(m))
        m = self.linear(m).squeeze()
        for examle in range(x.shape[0]):
            m[examle][token_nums[examle]:] = 0.0
        #print(m)
        mels_alignations_predicted = m

        if mels_alignations is None:
            mels_alignations = torch.exp(mels_alignations_predicted)
            for examle in range(x.shape[0]):
                mels_alignations[examle][token_nums[examle]:] = 0.0
            mels_gt_num = mels_alignations.sum(axis=1).long()

            print("mels alignations:", mels_alignations)
            print("mels gt num:", mels_gt_num)


        x_new = torch.zeros((x.shape[0], int(mels_gt_num.max().item()), x.shape[2])).to('cuda:0')
        for examle in range(x.shape[0]):
            cur_mel = 0
            cur_tgt = 0
            for i in range(x.shape[1]):
                # print(mels_alignations[examle][i])
                cur_tgt = cur_tgt + mels_alignations[examle][i].item()
                cur_tgt = min(cur_tgt, mels_gt_num[examle].item())
                while cur_mel < cur_tgt:
                    # print(mels_gt_num)
                    # print(cur_mel, cur_tgt)
                    x_new[examle][cur_mel] = x[examle][i]
                    cur_mel += 1

        return x_new, mels_alignations_predicted



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

        self.length_regulator = LengthRegulator(hidden_size=hidden_size)

        self.sec_position_enc = PositionalEncoding(
            emb_size=hidden_size, dropout=0.0
        )

        self.decoder_fft = TransformerBlock(hidden_size=hidden_size)

        self.linear_layer = nn.Linear(hidden_size, n_mels)

    def forward(self, tokens, token_nums, mels_alignations=None, mels_gt_num=None):
        x = self.embedding(tokens)
        x = self.fir_position_enc(x)
        x = self.encoder_fft(x)
        x, mels_alignations_predicted = self.length_regulator(
            x, token_nums, mels_alignations, mels_gt_num)
        x = self.sec_position_enc(x)
        x = self.decoder_fft(x)
        x = self.linear_layer(x)
        return x, mels_alignations_predicted