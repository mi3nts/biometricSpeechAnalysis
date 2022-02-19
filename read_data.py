# CODE TO READ EEG FILE COLLECTED WITH THE COGNIONICS MOBILE-128 SYSTEM USING MNE AND READ TRANSCRIPT FROM YOUTUBE VIDEO

# CODE AUTHORED BY: SHAWHIN TALEBI AND ARJUN SRIDHAR
# PROJECT: biometricSpeechAnalysis
# GitHub: https://github.com/mi3nts/biometricSpeechAnalysis
# ==============================================================================

# import libraries
import mne
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np

# INPUTS
#   - vhdr_fname = string. path to relevant .vhdr file
#   ~ example: vhdr_fname = "./data/2020_06_04_T05_U00T_EEG01.vhdr"

# OUTPUTS
#   - df_eeg_data = pandas dataframe with columns as biometric variables and rows as
#   timesteps
#   - df_eeg_data_youtube = pandas dataframe with columns as biometric variables and rows as
#   timesteps for when the youtube video was playing

# DEPENDENCIES
#   - none

# DEPENDERS
#   - none
def read_eeg(vhdr_fname):
    # define list of indicies for non-eeg channels
    misc_list = []
    for i in range(18):
        misc_list.append(i+64)

    # read raw data
    raw = mne.io.read_raw_brainvision(vhdr_fname, misc=misc_list, preload=True,
        verbose=False)
    raw.info['line_freq'] = 500.

    # Set montage
    montage = mne.channels.make_standard_montage('easycap-M1')
    raw.set_montage(montage, verbose=False)

    # Set common average reference
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # create pandas dataframe with eeg data
    df_eeg_data = pd.DataFrame(raw.get_data().transpose(), columns=raw.ch_names)
    
    # Trigger to sync with tobii eye data
    trig_tob = df_eeg_data.loc[df_eeg_data['TRIGGER'] == 3]
    start_tob = trig_tob.index[0]
    df_eeg_data = df_eeg_data.iloc[start_tob:, :]
    df_eeg_data.reset_index(inplace=True)
    
    # create time index - round to integer to match with transcript data
    times = list(range(len(df_eeg_data.index)))
    times = [int(t / 500) for t in times]
    df_eeg_data['Time'] = times
    
    # trigger for data during youtube video
    temp =  df_eeg_data.loc[df_eeg_data['TRIGGER'] == 8888]
    youtube_start = df_eeg_data.loc[df_eeg_data['TRIGGER'] == 8888].index[0]
    youtube_end = temp.index[len(temp.index) - 1]
    df_eeg_data_youtube = df_eeg_data.iloc[youtube_start:youtube_end + 1, :]
    print(df_eeg_data_youtube)
    
    df_eeg_data = df_eeg_data.loc[:, ~df_eeg_data.columns.isin(['T7', 'TRIGGER', 'ACC79', 'Packet Counter',
                                                              'ACC77', 'ACC78', 'AUX 2', 'AUX 1', 'index'])]
    
    df_eeg_data_youtube = df_eeg_data_youtube.loc[:, ~df_eeg_data_youtube.columns.isin(['T7', 'TRIGGER', 'ACC79', 
                                                                                        'Packet Counter', 'ACC77', 
                                                                                        'ACC78', 'AUX 2', 'AUX 1', 'index'])]
    
    # sync up time with youtube video
    df_eeg_data_youtube['Time'] = df_eeg_data_youtube['Time'] - df_eeg_data_youtube.iloc[0]['Time'] + 1
    df_eeg_data_youtube['Time'] = df_eeg_data_youtube['Time'].astype(int)
    
    return df_eeg_data, df_eeg_data_youtube

# INPUTS
#   - text file name = string. path to relevant youtube transcription video

# OUTPUTS
#   - df_text = pandas dataframe with columns as time (seconds) and words said at the corresponding time

# DEPENDENCIES
#   - none

# DEPENDERS
#   - none
def read_transcript_data(txt_file):
    # transcript info from video
    time_text = {'Time': [], 'words': []} 
    
    lines = open(txt_file).readlines()

    for i in range(0, len(lines) - 1, 2):
        time_sec = lines[i] # read timestamp
        text = lines[i + 1] # read text at timestamp
        
        # convert time to seconds
        time_sec = time_sec.strip()
        x = time.strptime(time_sec.split(',')[0],'%M:%S')
        sec = datetime.timedelta(minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
        
        time_text['Time'].append(int(sec))
        time_text['words'].append(text.strip())
    
    df_text = pd.DataFrame.from_dict(time_text)
    
    return df_text

df_eeg_data, df_eeg_data_youtube = read_eeg('./nlp_data/2022_01_14_T04_U002_EEG01/2022_01_14_T04_U002_EEG01.vhdr')
df_text = read_transcript_data('./nlp_data/daredevil_time.txt')
print(df_eeg_data_youtube)
print(df_text)
