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

# INPUTS
#   - vhdr_fname = string. path to relevant .vhdr file
#   ~ example: vhdr_fname = "./data/2020_06_04_T05_U00T_EEG01.vhdr"

# OUTPUTS
#   - eeg_data = pandas dataframe with columns as biometric variables and rows as
#   timesteps

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
    
    # create time index - round to integer to match with transcript data
    times = list(range(len(df_eeg_data.index)))
    times = [int(t / 500) for t in times]
    
    # take average for a given time
    df_eeg_data['Time'] = times
    df_eeg_data_avg = df_eeg_data.groupby('Time').mean()
    df_eeg_data_avg.reset_index(inplace=True)
    
    return df_eeg_data_avg

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

df_eeg_data = read_eeg('./nlp_data/2022_01_14_T04_U002_EEG01/2022_01_14_T04_U002_EEG01.vhdr')
df_text = read_transcript_data('./nlp_data/daredevil_time.txt')
print(df_eeg_data)
print(df_text)
