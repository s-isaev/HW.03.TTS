import torch
from itertools import islice
from torch.nn.modules import loss
from torch.utils.data import DataLoader

from dataset import MelSpectrogramConfig, MelSpectrogram
from dataset import LJSpeechDataset, LJSpeechCollator
from dataset import GraphemeAligner

from model import FastSpeech

def train(epochs=50):
    featurizer = MelSpectrogram(MelSpectrogramConfig())
    dataset = LJSpeechDataset('.')
    collator = LJSpeechCollator()
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=collator)

    dummy_batch = list(islice(dataloader, 1))[0]

    device = torch.device('cuda:0')
    aligner = GraphemeAligner().to(device)

    dummy_batch.durations = aligner(
        dummy_batch.waveform.to(device), 
        dummy_batch.waveforn_length, 
        dummy_batch.transcript
    )

    #print("Dummy batch:")
    #print(dummy_batch)

    model = FastSpeech(vocab_size=100, hidden_size=128, n_mels=80).to(device)

    tokens = dummy_batch.tokens.to(device)
    mels_gt = featurizer(dummy_batch.waveform).cuda().transpose(1,2)

    mels_gt_num = []
    for i in range(dummy_batch.waveform.shape[0]):
        wav = dummy_batch.waveform[i]
        wav_len = dummy_batch.waveforn_length[i]
        wav_len = wav_len.item()
        # print(wav.shape, wav_len)
        mels_gt_num.append(featurizer(wav[:wav_len]).shape[1])
    #print("Mel gt nums:", mels_gt_num)

    # print("Mels gt:", mels_gt)
    #print("GT shape:", mels_gt.shape)


    mels_alignations = dummy_batch.durations.to(device)
    mels_gt_num = torch.Tensor(mels_gt_num).to(device)
    token_nums = dummy_batch.token_lengths.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    loss_c = torch.nn.L1Loss().to(device)


    for i in range(epochs):
        print(i, end=': ')
        optimizer.zero_grad()
        mels = model(tokens, mels_alignations, mels_gt_num, token_nums)
        loss = loss_c(mels, mels_gt)
        loss.backward()
        optimizer.step()

        print(loss.item())

    return model, mels

    #print("Mels:")
    #print(mels)
    #print("Predicted shape:", mels.shape)