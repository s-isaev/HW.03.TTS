import torch
from itertools import islice
from torch._C import dtype
from torch.nn.modules import loss
from torch.utils.data import DataLoader

from dataset import MelSpectrogramConfig, MelSpectrogram
from dataset import LJSpeechDataset, LJSpeechCollator
from dataset import GraphemeAligner

from model import FastSpeech

from vocoder import Vocoder
import soundfile as sf
import scipy.io.wavfile

import tqdm
from copy import deepcopy

def train_batch(model, batch, aligner, featurizer, optimizer, loss_mel, loss_len, device):
    model.train()

    transcript_new = []
    for text in batch.transcript:
        text = text.replace("â", "")
        text = text.replace("\"", "")
        text = text.replace("é", "")
        text = text.replace("ü", "")
        text = text.replace("“", "")
        text = text.replace("”", "")
        text = text.replace("[", "")
        text = text.replace("]", "")
        transcript_new.append(text)
    batch.transcript = transcript_new

    batch.durations = aligner(
        batch.waveform.to(device), 
        batch.waveforn_length, 
        batch.transcript
    )
    tokens = batch.tokens.to(device)
    mels_gt = featurizer(batch.waveform).cuda().transpose(1,2)

    mels_gt_num = []
    for i in range(batch.waveform.shape[0]):
        wav = batch.waveform[i]
        wav_len = batch.waveforn_length[i]
        wav_len = wav_len.item()
        mels_gt_num.append(featurizer(wav[:wav_len]).shape[1])

    mels_alignations = batch.durations.to(device)
    mels_gt_num = torch.Tensor(mels_gt_num).to(device).long()
    mels_alignations = (mels_alignations.T/mels_alignations.sum(axis=1)*mels_gt_num).T
    token_nums = batch.token_lengths.to(device)

    optimizer.zero_grad()

    mels, mels_alignations_predicted, mels_num_predicted = \
        model(tokens, token_nums, mels_alignations, mels_gt_num)

    if tokens.shape != mels_alignations.shape:
        print(tokens.shape)
        print(mels_alignations.shape)
        print(batch.transcript)
        print(mels_alignations_predicted.shape)

    mels_alignations_log = mels_alignations
    for examle in range(tokens.shape[0]):
        mels_alignations_log[examle][token_nums[examle]:] = 1.0
        mels[examle][mels_gt_num[examle]:] = -10.0
        mels_gt[examle][mels_gt_num[examle]:] = -10.0
    mels_alignations_log = torch.log(mels_alignations_log)

    loss_mel_n = loss_mel(mels, mels_gt)
    loss_len_n = loss_len(mels_alignations_predicted, mels_alignations_log)

    mell = loss_mel_n.item()
    lenl = loss_len_n.item()

    loss = loss_mel_n + loss_len_n
    loss.backward()
    optimizer.step()

    return mels, {'mel': mell, 'len': lenl}


def prepare_model_loader_losses(device, datapath='.', batch_size=3):
    featurizer = MelSpectrogram(MelSpectrogramConfig())
    dataset = LJSpeechDataset(datapath)
    collator = LJSpeechCollator()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator)
    aligner = GraphemeAligner().to(device)
    model = FastSpeech(vocab_size=100, hidden_size=128, n_mels=80).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_mel = torch.nn.L1Loss().to(device)
    loss_len = torch.nn.MSELoss().to(device)

    return dataloader, model, aligner, featurizer, optimizer, loss_mel, loss_len

def train_checkpoint(steps=100, datapath='.'):
    device = torch.device('cuda:0')
    dataloader, model, aligner, featurizer, optimizer, loss_mel, loss_len = \
        prepare_model_loader_losses(
        device, datapath, batch_size=3
    )
    batch = list(islice(dataloader, 1))[0]

    mell = 0
    lenl = 0
    for i in range(steps):
        cbatch = deepcopy(batch)
        mels, losses = train_batch(
            model, cbatch, aligner, featurizer, optimizer, loss_mel, loss_len, device
        )

        mell += losses['mel']
        lenl += losses['len']
        if (i + 1) % 100 == 0:
            print("Step:", i+1, end=' ')
            print("Mel:", mell/100, end=' ')
            print("Len:", lenl/100)
            mell = 0
            lenl = 0

    return model, mels


def train(epochs=50, datapath='.', batch_size=3):
    device = torch.device('cuda:0')
    dataloader, model, aligner, featurizer, optimizer, loss_mel, loss_len = \
        prepare_model_loader_losses(
        device, datapath, batch_size
    )

    for epoch in range(epochs):
        i = 0
        mell = 0
        lenl = 0
        for batch in tqdm.tqdm(dataloader):
            mels, losses = train_batch(
                model, batch, aligner, featurizer, optimizer, loss_mel, loss_len, device
            )

            mell += losses['mel']
            lenl += losses['len']
            if (i + 1) % 100 == 0:
                print("Step:", i + 1, end=' ')
                print("Mel:", mell/100, end=' ')
                print("Len:", lenl/100)
                mell = 0
                lenl = 0
            i += 1

    return model, mels

def eval(model, device, datapath='.'):
    featurizer = MelSpectrogram(MelSpectrogramConfig())
    dataset = LJSpeechDataset(datapath)
    collator = LJSpeechCollator()
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=collator)

    dummy_batch = list(islice(dataloader, 1))[0]

    aligner = GraphemeAligner().to(device)

    dummy_batch.durations = aligner(
        dummy_batch.waveform.to(device), 
        dummy_batch.waveforn_length, 
        dummy_batch.transcript
    )

    tokens = dummy_batch.tokens.to(device)
    mels_gt = featurizer(dummy_batch.waveform).cuda().transpose(1,2)

    mels_alignations = dummy_batch.durations.to(device)
    token_nums = dummy_batch.token_lengths.to(device)

    loss_mel = torch.nn.L1Loss().to(device)
    loss_len = torch.nn.MSELoss().to(device) 


    with torch.no_grad():
        mels, mels_alignations_predicted, mels_num_predicted = model(tokens, token_nums)

        # loss_mel_n = loss_mel(mels, mels_gt)
        loss_len_n = loss_len(mels_alignations_predicted, mels_alignations)
        # print("Mel:", loss_mel_n.item(), end=' ')
        print("Len:", loss_len_n.item())

    vocoder = Vocoder().to(device).eval()
    for i in range(mels.shape[0]):
        mel = mels[i:i+1,:mels_num_predicted[i]].transpose(1,2)
        wav = vocoder.inference(mel).cpu().numpy()
        print(wav[0])
        scipy.io.wavfile.write(str(i) + '.wav',22050, wav[0])