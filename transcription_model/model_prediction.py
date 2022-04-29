"""
Performs prediction using checkpoint of model
"""
import torch
from model_utils import EEGDataset, MINTSModel, ToTensor, GreedyCTCDecoder
import torchaudio
import pandas as pd

device = torch.device('cpu')

wav2vec_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H # torchaudio.pipelines.WAV2VEC2_BASE

model = wav2vec_bundle.get_model().to(device)

dataset = EEGDataset('data/BadTalk.wav', 'data/nlp_data/2022_01_14_T04_U002_EEG01/2022_01_14_T04_U002_EEG01.vhdr', 
            'data/nlp_data/2022_01_14_T04_U002_Tobii01/segments/1/segment.json', 
            'data/nlp_data/2022_01_14_T04_U002_Tobii01/segments/1/livedata.json.gz',
            'data/nlp_data/daredevil_time.txt',
            labels=wav2vec_bundle.get_labels(),
            transforms=[ToTensor()],
            batch_size=128000)
waveform, sample_rate = torchaudio.load('data/BadTalk.wav')
waveform = waveform.to(device)
if sample_rate != wav2vec_bundle.sample_rate:
    print(f'Input sample rate was {sample_rate}, resampling to {wav2vec_bundle.sample_rate}')
    waveform = torchaudio.functional.resample(waveform, sample_rate, wav2vec_bundle.sample_rate)
mints_model = MINTSModel(batch_size=128000)
decoder = GreedyCTCDecoder(labels=wav2vec_bundle.get_labels())

mints_model.load_state_dict(torch.load('checkpoint_2000.pt')['model_state_dict'])
mints_model.eval()

output = []
batches = len(dataset) // 128000
with torch.no_grad():
    for i in range(batches):
        batch = dataset[(128000 * i):(128000 * (i + 1))]
        logits = mints_model(batch)
        emission, _ = model(batch['audio'])
        output.append([i, decoder(logits), decoder(emission[0])])
        
output_df = pd.DataFrame(output, columns=['batch', 'model', 'wav2vec'])
output_df.to_csv('transcription_model.csv')