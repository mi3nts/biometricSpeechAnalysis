# CODE TO READ EEG FILE COLLECTED WITH THE COGNIONICS MOBILE-128 SYSTEM USING MNE, TOBII EYE TRACKING DATA, AND TRANSCRIPT FROM YOUTUBE VIDEO

# CODE AUTHORED BY: SHAWHIN TALEBI, OMAR LUNA, AND ARJUN SRIDHAR
# PROJECT: biometricSpeechAnalysis
# GitHub: https://github.com/mi3nts/biometricSpeechAnalysis
# ==============================================================================

# import libraries
import mne
import pandas as pd
import datetime
import time
import json as js
from datetime import datetime as dt
import gzip

# INPUTS
#   -participant_filename: participant json file from tobii device
#   -data_filename: live data json filefrom tobii device
# OUTPUTS
#   - df_tobii: pandas DataFrame containing columns for the left and right pupil diameter
#               as well as the avg diameter of the left and right pupil diameter

# Function that returns pupil diameter data from a json file as a dataframe
def read_tobii(participant_filename, data_filename):
    
    # function returns initial timestamp from participant.json file
    def get_ptimestamp(filename):
        with open(filename) as f:
          for i, line in enumerate(f):
    
            # find index of substring containing pa_created
            ss_index = line.find('seg_t_start')
            if ss_index > -1:
    
              # parse the timestamp and convert to datetime
              ts = line[ss_index+14: len(line)-2]
              ts = ts[1: len(ts) - 1]
              dt_timestamp = dt.strptime(ts, "%Y-%m-%dT%H:%M:%S+%f" )
        return dt_timestamp
    
    def data_parsing(dict_data):
        # searches for pd key in dictionary
        if "pd" in dict_data:
    
          # removes keys not used and returns dictionary
          del dict_data["s"], dict_data["gidx"],
          return dict_data
    
    # Open file with data values and store them in list
    def get_data(filename, list_data):
        with gzip.open(filename) as f_livedata:
            for i, line in enumerate(f_livedata):
              p_data = data_parsing(js.loads(line))
              if p_data!=None:
                list_data.append(p_data)
              else:
                continue
    
    
    # list that contains pd data
    list_data = []
    # list that contains final datetime values
    l_final_ts = []
    
    # stores pd data in list_data from file
    get_data(data_filename, list_data)
    
    # returns initial timestamp from file
    p_ts = get_ptimestamp(participant_filename)
    # Creates initial dataframe
    df_pd = pd.DataFrame.from_dict(list_data)
    df_pd = df_pd.pivot(index='ts', columns= 'eye')
    
    # Converts initial datetime value to int
    timestamp = int(round(p_ts.timestamp()))
    # stores index values
    index_vals = df_pd.index.values
    # calculates the difference in seconds between values
    step = 1/100
    
    # Converts int values to datetime values
    for i in range(0, index_vals.size, 1):
        l_final_ts.append(dt.fromtimestamp(timestamp + (i*step)))
    
    # Final dataframe
    df_final = pd.DataFrame(df_pd.values, columns= df_pd.columns,  index=l_final_ts)
    # Create mean column and append to final dataframe
    l_avg = [0] * len(df_final.index)
    for i, p_dia in enumerate(df_final.values):
        l_avg[i] = (p_dia[0] + p_dia[1]) / 2
    df_final['pd_avg'] = l_avg
      
    times = list(range(len(df_final.index)))
    times = [int(t / 100) for t in times]
    df_final['Time'] = times
    df_final['Timestamp'] = l_final_ts
    df_final.columns = ['Pupil_Left', 'Pupil_Right', 'Pupil_Avg', 'Time', 'Timestamp']
    df_final.index.name = None
    df_final.reset_index(inplace=True)
  
    df_tobii = df_final[['Time', 'Pupil_Left', 'Pupil_Right', 'Pupil_Avg', 'Timestamp']]
  
    return df_tobii

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
    
    df_eeg_data = df_eeg_data.loc[:, ~df_eeg_data.columns.isin(['T7', 'TRIGGER', 'ACC79', 'Packet Counter',
                                                              'ACC77', 'ACC78', 'AUX 2', 'AUX 1', 'index'])]
    
    df_eeg_data_youtube = df_eeg_data_youtube.loc[:, ~df_eeg_data_youtube.columns.isin(['T7', 'TRIGGER', 'ACC79', 
                                                                                        'Packet Counter', 'ACC77', 
                                                                                        'ACC78', 'AUX 2', 'AUX 1', 'index'])]
    
    # sync up time with youtube video
    #df_eeg_data_youtube['Time'] = df_eeg_data_youtube['Time'] - df_eeg_data_youtube.iloc[0]['Time'] + 1
    #df_eeg_data_youtube['Time'] = df_eeg_data_youtube['Time'].astype(int)
    
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

# INPUTS
#   - eeg file name = string. path to relevant .vhdr file
#   - participant file name = participant json file from tobii device
#   - live data file name = live data json filefrom tobii device
#   - text file name = string. path to relevant youtube transcription video

# OUTPUTS
#   - df_eeg_data = dataframe with biometric data
#   - df_tobii = dataframe with tobii data
#   - df_eeg_data_youtube = dataframe with biometic data for youtube video duration
#   - df_tobii_youtube = dataframe with tobii data for youtube video duration
#   - df_text = pandas dataframe with columns as time (seconds) and words said at the corresponding time from youtube video

# DEPENDENCIES
#   - read_eeg(vhdr_fname)
#   - read_tobii(participant_filename, data_filename)
#   - read_transcript_data(txt_file)

# DEPENDERS
#   - none
def read_all_data_youtube(eeg_file, participant_file, livedata_file, transcript_file): 
    df_eeg_data, df_eeg_data_youtube = read_eeg(eeg_file)
    df_tobii = read_tobii(participant_file, 
                           livedata_file)
    
    youtube_start = df_eeg_data_youtube.iloc[0]['Time']
    youtube_end = df_eeg_data_youtube.iloc[-1]['Time']
    
    # get tobii data for youtube video
    df_tobii_youtube_start = df_tobii.loc[df_tobii['Time'] == youtube_start]
    df_tobii_youtube_end = df_tobii.loc[df_tobii['Time'] == youtube_end]
    df_tobii_youtube = df_tobii.iloc[df_tobii_youtube_start.index[0]: df_tobii_youtube_end.index[-1] + 1, :]
    
    # sync times with youtube video
    df_eeg_data_youtube['Time'] = df_eeg_data_youtube['Time'] - df_eeg_data_youtube.iloc[0]['Time'] + 1
    df_eeg_data_youtube['Time'] = df_eeg_data_youtube['Time'].astype(int)
    
    df_tobii_youtube['Time'] = df_tobii_youtube['Time'] - df_tobii_youtube.iloc[0]['Time'] + 1
    df_tobii_youtube['Time'] = df_tobii_youtube['Time'].astype(int)
                                 
    df_text = read_transcript_data(transcript_file)
    
    return df_eeg_data, df_tobii, df_eeg_data_youtube, df_tobii_youtube, df_text
    

data = read_all_data_youtube('./nlp_data/2022_01_14_T04_U002_EEG01/2022_01_14_T04_U002_EEG01.vhdr', 
          './nlp_data/2022_01_14_T04_U002_Tobii01/segments/1/segment.json', 
          './nlp_data/2022_01_14_T04_U002_Tobii01/segments/1/livedata.json.gz',
          './nlp_data/daredevil_time.txt')

df_eeg_data, df_tobii, df_eeg_data_youtube, df_tobii_youtube, df_text = data[0], data[1], data[2], data[3], data[4]
print(df_tobii_youtube)
print(df_eeg_data_youtube)
