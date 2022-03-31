import librosa
import soundfile as sf
import os
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2ForCTC
from read_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_batches(batch_size=250000, new_sample_rate=16000, outdir="daredevil16k", infile="audiototext/Daredevil and Punisher Argument.wav"):
    # Split audio file into batches of 250k samples (15.625 sec), with sample rate 16k
    # This length are chosen based on how the wav2vec2 base model was trained
    os.mkdir(outdir)
    # Use librosa to change sample rate
    audio_input, sample_rate = librosa.load(infile, sr=new_sample_rate)
    for i_sample in range(len(audio_input)//batch_size):
        start = i_sample*batch_size
        end = start + batch_size
        sf.write(f"{outdir}/{i_sample}.wav", audio_input[start:end], sample_rate)


def get_waveforms():
    # Retrieve the raw waveforms from disk
    for i_batch in range(10):
        filename = f"daredevil16k/{i_batch}.wav"
        waveform, sample_rate = sf.read(filename)
        yield waveform


def get_acoustic_features(waveforms):
    # Takes a tuple/list of arrays, each representing a raw waveform
    # Returns values of last hidden layer of the raw pretrained Wav2Vec2 model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    for waveform in waveforms:
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        yield outputs.last_hidden_state


def get_bio_input(i_batch):
    # For now, this function just returns the heart rate data corresponding to the given batch number
    df_eeg_data, df_eeg_data_youtube = read_eeg("nlp_data/2022_01_14_T04_U002_EEG01/2022_01_14_T04_U002_EEG01.vhdr")
    # Get every nth row of heart rate data, so that we have 781 samples

    df = df_eeg_data_youtube
    # TODO: use timestamps instead of youtube time
    # TODO: Why does the time start at 22
    
    start = 15.625*i_batch+22
    end = start + 15.625
    df = df[(start <= df["Time"]) & (df["Time"] < end)]
    stride = len(df)//781
    heart_rate = df[:stride*781:stride]['HR']
    heart_rate = heart_rate.to_numpy().reshape((1,781,1))
    return heart_rate


def get_targets(waveforms):
    # Given a list/tuple of arrays, return the sequence that Wav2Vec2 would have predicted for each
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    inputs = get_waveforms()
    for input in inputs:
        input = processor(input, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**input).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        yield processor.batch_decode(predicted_ids)


if __name__ == "__main__":
    if not os.path.exists("daredevil16k/"):
        create_batches()
    waveforms = list(get_waveforms())
    print([waveform.shape for waveform in waveforms])
    print([f.shape for f in get_acoustic_features(waveforms)])
    print([get_bio_input(i).shape for i in range(10)])
    print(*tuple(get_targets(waveforms)), sep="\n")