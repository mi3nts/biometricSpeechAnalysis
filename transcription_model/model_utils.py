from scipy import signal
import sklearn
from sklearn.metrics import balanced_accuracy_score
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.nn import functional as F
import pytorch_lightning as pl
from read_data import read_all_data_youtube

class EEGDataset(torch.utils.data.Dataset):
    """MINTS EEG Dataset"""
    eeg_sample_ratio = 500

    def __init__(self, audio_file, eeg_file, participant_file, livedata_file, transcript_file, labels, transforms=None, batch_size=64000) -> None:
        super().__init__()
        df_eeg_data, df_tobii, df_eeg_data_youtube, df_tobii_youtube, df_text = \
            read_all_data_youtube(eeg_file, participant_file, livedata_file, transcript_file)
        # label_type = pd.api.types.CategoricalDtype(categories=labels, ordered=True)
        self.batch_size = batch_size
        self.tobii_data = df_tobii_youtube
        self.transcript_data = df_text
        self.transcript_data.loc[-1] = [0, ' ']
        self.transcript_data.index = self.transcript_data.index + 1
        self.transcript_data = self.transcript_data.sort_index()
        self.transcript_data['Time'] *= 16000  # Upsampling
        self.transcript_data['words'] = self.transcript_data['words'].str.upper().replace('\s+', '|', regex=True)
        self.transcript_data['words'] = self.transcript_data['words'].str.ljust(40, '|')
        self.labels = labels
        print(labels)

        self.transcript_data['encoded'] = self.transcript_data['words'].apply(self._one_hot)
        print(self.transcript_data['words'].iloc[0])
        print(self.transcript_data['encoded'].iloc[0:10])
        print(np.array(self.transcript_data['encoded'].iloc[0:10].tolist()).shape)
        # print('FIRST WORD:')
        # print(self.transcript_data['words'].iloc[0])
        # print('FIRST WORD SPLIT:')
        # self.transcript_data['words'] = self.transcript_data['words'].str.rstrip().str.split('')
        # print(self.transcript_data['words'].iloc[0])
        # print('FIRST WORD CATS:')
        # print(self.transcript_data['words'].apply(lambda x: pd.Series(list(x)[1:-1])).astype(label_type).iloc[0])
        # print(pd.get_dummies(self.transcript_data['words'].apply(lambda x: pd.Series(list(x)[1:-1])).astype(label_type))[0])

        self.transcript_data['TimeDiff'] = self.transcript_data['Time'].diff(periods=-1).fillna(0) * -1
        self.transcript_data = self.transcript_data.loc[self.transcript_data.index.repeat(self.transcript_data['TimeDiff'])].reset_index(drop=True)
        self.transcript_data.drop('TimeDiff', axis=1, inplace=True)
        self.audio_data, self.sample_rate = torchaudio.load(audio_file)
        self.audio_data = torchaudio.functional.resample(self.audio_data, self.sample_rate, 16000)
        eeg_data = signal.resample(df_eeg_data_youtube.to_numpy(), self.audio_data.shape[1])
        self.eeg_data = pd.DataFrame(eeg_data, columns=df_eeg_data.columns)
        self.transforms = transforms

    def _one_hot(self, row):
        output = np.zeros((len(row), len(self.labels)))
        for i in range(len(row)):
            output[i, self.labels.index(row[i])] = 1
        return output

    def __len__(self):
        return self.audio_data.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Use idx as "start" of sequence and get entire sequence as an item
        # This allows for random shuffle dataloader while keeping sequential items
        eeg_data = self.eeg_data.iloc[idx:(idx + self.batch_size)]
        # tobii_data = self.tobii_data.iloc[idx]
        transcript_data = self.transcript_data['encoded'].iloc[idx:(idx + self.batch_size)]
        audio_data = self.audio_data[:, idx:(idx + self.batch_size)]

        sample = {
            'eeg': np.array(eeg_data),
            # 'tobii': torch.from_numpy(np.array(tobii_data)),
            'audio': audio_data,
            'text': torch.from_numpy(np.array(transcript_data.tolist()))
        }
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)
        return sample

class BatchSeqSampler(torch.utils.data.sampler.Sampler):
    """THIS DOESN'T WORK YET"""
    def __init__(self, data_source, num_samples=64000):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        n = len(self.data_source) - self.num_samples
        start_n = np.random.randint(0, n)
        return iter(range(start_n, start_n + self.num_samples))

    def __len__(self):
        return self.num_samples

class Spectrogram(object):
    """Gets power spectrum of EEG voltages"""

    def __call__(self, sample):
        # sample['eeg'] = self.psd(sample['eeg'])
        # TODO: Treat Oz, PpO, etc (not-EEG) separately
        sample['eeg'] = np.apply_along_axis(lambda x: signal.welch(x, 16000, scaling='spectrum')[1], 0, sample['eeg'])
        return sample

class ToTensor(object):
    """Converts ndarrays to Tensors"""

    def __call__(self, sample):
        sample['eeg'] = torch.from_numpy(sample['eeg'])
        return sample

class MINTSModel(pl.LightningModule):#(torch.nn.Module):
    """ML Model"""
    def __init__(self, batch_size) -> None:
        seq_size = 40
        super(MINTSModel, self).__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.w2v_model = bundle.get_model()
        for p in self.w2v_model.parameters():
            p.requires_grad = False
        self.audio_model = torch.nn.Sequential(
            # torch.nn.ReLU(),
            torch.nn.Linear(2 * 399 * 768, seq_size * 29)  # ISSUE: Big reduction in dimensionality
        )
        self.eeg_model = torch.nn.Sequential(
            # torch.nn.Conv1d(batch_size, 2048, 4, 16),
            # torch.nn.ReLU(),
            torch.nn.Conv1d(batch_size, seq_size * 29, 2)
            # torch.nn.Conv1d(batch_size, seq_size * 29, 2)
        )
        self.eeg_output = torch.nn.Linear(seq_size * 29 * 4, seq_size * 29)
        self.feature_output = torch.nn.Linear(seq_size * 2 * 29, seq_size * 29)
        self.logits = torch.nn.Sequential(
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"Audio Input shape: {x['audio'].shape}")
        audio_features, _ = self.w2v_model.extract_features(x['audio'])
        audio_final = self.audio_model(audio_features[-1].flatten())
        # print(f'Audio Model Output shape: {audio_final.shape}')
        # input_shape = x['eeg'].shape
        # print(f'EEG Input shape: {input_shape}')

        eeg_output = self.eeg_model(x['eeg'].float())
        # print(f'EEG Output 1 shape: {eeg_output.shape}')
        eeg_output = self.eeg_output(eeg_output.flatten())
        # print(f'EEG Output 2 shape: {eeg_output.shape}')
        final_features = torch.cat([audio_final, eeg_output])
        # print(f'Final Hidden shape: {final_features.shape}')
        feature_output = self.feature_output(final_features).reshape(40, 29)
        # print(f'Final Output shape: {feature_output.shape}')
        final_output = self.logits(feature_output)
        # print(f'Logits shape: {final_output.shape}')
        return final_output

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = {
            'text': torch.squeeze(train_batch['text']),
            'eeg': torch.squeeze(train_batch['eeg']),
            'audio': torch.squeeze(train_batch['audio'])
        }
        logits = self.forward(x)
        # PICK MODE OF BATCH TARGET
        # TEST AVERAGE AS WELL
        common_label = torch.mode(train_batch['text'])
        target = torch.argmax(common_label, dim=-1)  # Get expected target character indices
        loss = F.cross_entropy(logits, target.flatten())  # Compute loss using logits and target indices
        self.log('train_loss', loss)
        return loss
    
    # def validation_step(self, val_batch, batch_idx):
    #     logits = self.forward(val_batch)
    #     # PICK MODE OF BATCH TARGET
    #     # TEST AVERAGE AS WELL
    #     common_label = torch.mode(val_batch['text'])
    #     target = torch.argmax(common_label, dim=-1)  # Get expected target character indices
    #     loss = F.cross_entropy(logits, target.flatten())  # Compute loss using logits and target indices
    #     self.log('val_loss', loss)