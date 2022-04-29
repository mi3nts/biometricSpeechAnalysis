
import os
import matplotlib.pyplot as plt
from random import random, sample
import torch
import torchaudio
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import random_split
import time
from model_utils import BatchSeqSampler, EEGDataset, MINTSModel, Spectrogram, ToTensor

class GreedyCTCDecoder(torch.nn.Module):
    """Maps logits/emissions into letters"""
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """
        Greedily (argmax) chooses string of labels given sequence emission
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

print(torch.__version__)
print(torchaudio.__version__)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)

wav2vec_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H # torchaudio.pipelines.WAV2VEC2_BASE
print(f'Expected Wave2Vec Sample Rate: {wav2vec_bundle.sample_rate}')

model = wav2vec_bundle.get_model().to(device)
print(model.__class__)

dataset = EEGDataset('data/BadTalk.wav', 'data/nlp_data/2022_01_14_T04_U002_EEG01/2022_01_14_T04_U002_EEG01.vhdr', 
            'data/nlp_data/2022_01_14_T04_U002_Tobii01/segments/1/segment.json', 
            'data/nlp_data/2022_01_14_T04_U002_Tobii01/segments/1/livedata.json.gz',
            'data/nlp_data/daredevil_time.txt',
            labels=wav2vec_bundle.get_labels(),
            transforms=[ToTensor()],
            batch_size=128000)
print(dataset.eeg_data['Time'].describe())
waveform, sample_rate = torchaudio.load('data/BadTalk.wav')
waveform = waveform.to(device)
if sample_rate != wav2vec_bundle.sample_rate:
    print(f'Input sample rate was {sample_rate}, resampling to {wav2vec_bundle.sample_rate}')
    waveform = torchaudio.functional.resample(waveform, sample_rate, wav2vec_bundle.sample_rate)
# # Crop waveform because its huge and idc
print(f'Original Waveform shape: {waveform.shape}')
mints_model = MINTSModel(batch_size=128000)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=4)
trainer = pl.Trainer(accelerator='cpu', strategy='ddp', precision='bf16', limit_train_batches=0.1)
trainer.fit(mints_model, train_loader)
# optimizer = torch.optim.SGD(mints_model.parameters(), lr=0.05)
# criterion = torch.nn.NLLLoss()
# decoder = GreedyCTCDecoder(labels=wav2vec_bundle.get_labels())
# batch_size = 128000
# total_loss = 0
# train_start = time.time()
# epochs = int(60 * 60 * 6 / 9)
# print(f'Training for {epochs} epochs.')
# for i in range(epochs):
#     random_index = np.random.randint(0, len(dataset) - batch_size)
#     batch = dataset[random_index:(random_index + batch_size)]
#     logits = mints_model(batch)
#     # PICK MODE OF BATCH TARGET
#     # TEST AVERAGE AS WELL
#     common_label, label_indices = torch.mode(batch['text'], dim=0)
#     target = torch.argmax(common_label, dim=-1)  # Get expected target character indices
#     loss = criterion(logits, target.flatten())  # Compute loss using logits and target indices

#     # CREATE LEARNING RATE SCHEDULER
#     # INTEGRATE TENSORBOARD/VISUALIZATIONS (MAYBE CREATE BASIC .CSV OF MODEL STATS)
#     optimizer.zero_grad()
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#     optimizer.step()
#     step_loss = loss.item()
#     if i % 100 == 0:
#         print(f'Iteration {i} with Loss: {step_loss} and average loss: {total_loss / (i + 1)}')
#     if i % 1000 == 0:
#         # Save a checkpoint
#         torch.save({
#             'epoch': i,
#             'model_state_dict': mints_model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss
#         }, f'checkpoint_{i}.pt')
#     total_loss += step_loss
# print(f'Training took {time.time() - train_start} seconds, or {(time.time() - train_start) / epochs} s/epoch.')
# torch.save(mints_model.state_dict(), 'model_state.pt')
# # For inference need to do model = torch.load(PATH) and then model.eval()
# with torch.inference_mode():
#     for i in range(5):
#         random_index = np.random.randint(0, len(dataset) - batch_size)
#         batch = dataset[random_index:(random_index + batch_size)]
#         logits = mints_model(batch)
#         print(logits)  # Infer logits given this batch
#         print(f"Predicted Transcript: {decoder(logits)}\tActual Transcript: {decoder(batch['text'][0])}")

    # with torch.inference_mode():
    #     print(batch['audio'].shape)
    #     emission_orig, _ = model(waveform[:, 0:batch_size])
    #     emission_model, _ = model(batch['audio'])
    #     print(emission_orig.shape)
    #     print(emission_model.shape)
    #     plt.imshow(emission_orig[0].cpu().T)
    #     plt.title("Classification result")
    #     plt.xlabel("Frame (time-axis)")
    #     plt.ylabel("Class")
    #     plt.show()
    #     print("Class labels:", wav2vec_bundle.get_labels())
    #     decoder = GreedyCTCDecoder(labels=wav2vec_bundle.get_labels())
    #     transcript = decoder(emission_orig[0])
    #     print(f'Original Transcript:\n{transcript}\n')
    #     transcript = decoder(emission_model[0])
    #     print(f'New Transcript:\n{transcript}\n')

# waveform_crop = torch.split(waveform, sample_rate, dim=1)[0]
# print(f'New Waveform shape: {waveform_crop.shape}')
# batch_size = 128000
# dataloader = torch.utils.data.DataLoader(dataset, sampler=BatchSeqSampler(dataset, num_samples=batch_size), num_workers=0, batch_size=batch_size)
# # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
# for i_batch, sample_batch in enumerate(dataloader):
#     print('BATCH: ', i_batch)
#     if i_batch == 0:
#         with torch.inference_mode():
#         #     features, _ = model.extract_features(waveform_crop)
#         #     fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
#         #     for i, feats in enumerate(features):
#         #         ax[i].imshow(feats[0].cpu())
#         #         ax[i].set_title(f"Feature from transformer layer {i+1}")
#         #         ax[i].set_xlabel("Feature dimension")
#         #         ax[i].set_ylabel("Frame (time-axis)")
#         #     plt.tight_layout()
#         #     plt.show()
#             print(sample_batch['audio'].shape)
#             batch = torch.squeeze(sample_batch['audio']).T
#             print(batch.shape)
#             emission_orig, _ = model(waveform[:, 0:64000])
#             emission_model, _ = model(batch)
#             print(emission_orig.shape)
#             print(emission_model.shape)
#             plt.imshow(emission_orig[0].cpu().T)
#             plt.title("Classification result")
#             plt.xlabel("Frame (time-axis)")
#             plt.ylabel("Class")
#             plt.show()
#             print("Class labels:", wav2vec_bundle.get_labels())
#             decoder = GreedyCTCDecoder(labels=wav2vec_bundle.get_labels())
#             transcript = decoder(emission_orig[0])
#             print(f'Original Transcript:\n{transcript}\n')
#             transcript = decoder(emission_model[0])
#             print(f'New Transcript:\n{transcript}\n')
#         break