# -*- coding: utf-8 -*-
"""REAL MINTS March 26 Deliverables.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nGuMVrvAVbVV_ODxyO1mOdgAR7FHm-DZ
"""
import argparse

ap = argparse.ArgumentParser(description="Visualize EEG data")

ap.add_argument("zip_password", help="The password for the MINTS zip (not included for security reasons)")

args = ap.parse_args()


# Prepare environment
import os
import subprocess
import re

'''
  !pip install spacy dash jupyter-dash pandas==1.2.0 mne[data] gdown yt-dlp webvtt-py librosa > /dev/null
  !python -m spacy download en_core_web_sm
  !apt install p7zip -y > /dev/null
  !echo installed >> _installed
'''

# Common imports
import sys, os, time

# Module imports
import pandas as pd
import spacy as sp
import mne
import numpy as np # We might need it.

# NLTK
import nltk
import threading

# Caption processing
import webvtt

# We can just download this asynchronously
nltk.download('vader_lexicon')
nltk_thread = threading.Thread(target=lambda: nltk.download("popular", quiet=True))
nltk_thread.setDaemon(True)
nltk_thread.start()

from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag

# Matplotlib (alternative to dash for simple debugging graphs)
import matplotlib.pyplot as plt
# Enable interactive mode
plt.ion()

# Download and decrypt MINTS data. I think I put the 3 in the wrong place in the
# password. Oh well.
# Also, it seems like the gdown started randomly failing at some point during
# the week, so I'm temporarily hosting it on a temporary web server. I'm sorry
# in advance for my terrible upload bandwidth.

# Import dash selectively
from dash import dash_table
import dash
from dash import html, dcc

try:
    # Test if we're in a a Colab environment
    import google.colab 
    from jupyter_dash import JupyterDash
    class Dash(JupyterDash):
        def run_server(self, *args, **kwargs):
            return JupyterDash.run_server(self, *args, mode="inline", debug=True, **kwargs)
except ImportError:
    from dash import Dash

import requests
from zipfile import ZipFile
import os
from io import BytesIO

to_open = "LargeMINTS.zip" # Replace with LargeMINTS.zip for the large dataset
nlp_data = "large_nlp_data" if "Large" in to_open else "nlp_data"

if not os.path.exists(os.path.join(nlp_data, "./2022_01_14_T04_U002_EEG01/2022_01_14_T04_U002_EEG01.vhdr")):
    zip_password = args.zip_password

    root_dir = os.path.split(os.path.abspath("."))[0]
    zip_path = os.path.join(root_dir, "data", to_open)
    os.makedirs(nlp_data, exist_ok=True)

    print("Attempting to call 7zip")
    subprocess.check_call(["7z", "-y", "x", "-p" + zip_password, zip_path], cwd=os.path.abspath(nlp_data))

# We assume the biometric model is one directory up
sys.path.append(os.path.split(os.path.abspath("."))[0])

# Import them
# from read_data import *

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
    #youtube_start = df_eeg_data.loc[df_eeg_data['TRIGGER'] == 8888].index[0]
    youtube_end = temp.index[len(temp.index) - 1]
    df_eeg_data = df_eeg_data.iloc[:youtube_end + 1, :]
    #print(df_eeg_data_youtube)
    
    df_eeg_data = df_eeg_data.loc[:, ~df_eeg_data.columns.isin(['T7', 'TRIGGER', 'ACC79', 'Packet Counter',
                                                              'ACC77', 'ACC78', 'AUX 2', 'AUX 1', 'index'])]
    
    # sync up time with youtube video
    #df_eeg_data_youtube['Time'] = df_eeg_data_youtube['Time'] - df_eeg_data_youtube.iloc[0]['Time'] + 1
    #df_eeg_data_youtube['Time'] = df_eeg_data_youtube['Time'].astype(int)
    
    return df_eeg_data

df_eeg_data = read_eeg(os.path.join(nlp_data, "2022_01_14_T04_U002_EEG01/2022_01_14_T04_U002_EEG01.vhdr"))

yt_wav = "BadTalk.wav"
if not os.path.exists(os.path.join(nlp_data, "BadTalk.wav.en.vtt")):
    print("Attempting to call Youtube-DL Plus (yt-dlp)")
    subprocess.check_call(f"yt-dlp -x --audio-format wav --audio-quality 0 --write-auto-sub -o {yt_wav} https://www.youtube.com/watch?v=nGS8_R79vls", cwd=nlp_data)

samplerate = 500

vtt = webvtt.read(os.path.join(nlp_data, yt_wav + ".en.vtt"))

def ts_to_frameno(ts):
  # Converts a 00:00:00.000 timestamp into a frame number
  hrs, mins, secs = ts.split(":")
  total = int(hrs) * 3600 + int(mins) * 60 + float(secs)
  frameno = int(total * samplerate)
  return frameno

cap_re = re.compile(r"<(?P<time>\d{2}:\d{2}:\d+\.\d+)><c>(?P<text>[^<]+)</c>")

# Clearly, not the most efficient way of doing it, but it works
frame_to_caption = [0] * len(df_eeg_data)
caption_text = ['']

for caption in vtt:
  # print(caption)
  frame_start = ts_to_frameno(caption.start)
  frame_end = ts_to_frameno(caption.end)
  words = []

  for word_start, word_text in cap_re.findall(caption.raw_text):
    word_start = ts_to_frameno(word_start)
    word_text = word_text.strip()
    if not word_text: continue

    words.append((word_start, word_text))
  
  bad_guess = False
  if len(words) == 0 and caption.text.strip():
    words.append((frame_start, caption.text.strip()))
    bad_guess = True

  for i, (word_start, word_text) in enumerate(words):
    if i == len(words) - 1:
      word_end = frame_end
    else:
      word_end = words[i + 1][0] - 1
    
    for i in range(frame_start, min(len(df_eeg_data), word_end + 1)):
      frame_to_caption[i] = len(caption_text) if (frame_to_caption[i] == 0 or not bad_guess) else frame_to_caption[i]
    caption_text.append(word_text)

#print(sum(frame_to_caption))
df_eeg_data["Caption"] = frame_to_caption
del frame_to_caption

fftsize = 512
sampleduration = 1 / samplerate * fftsize

fakefreq = 20
fakedata = {"T": [np.cos(np.pi * 2 / sampleduration / fftsize * fakefreq * t) for t in range(fftsize * 4)]}

def do_fft(data, label):
  start = time.time()
  fft = [1 / fftsize * np.fft.fft(data[label][i:i+fftsize]) for i in range(len(data[label]) - fftsize)] + [[0] * fftsize] * fftsize
  # Create tolerance threshold
  threshold = np.max(np.abs(fft)) / 10000
  fft2 = np.where(np.abs(fft) >= threshold, fft, 0)
  data[label + "phase"] = [np.arctan2(np.imag(X), np.real(X)) * 180 / np.pi for X in fft2]
  del fft2
  data[label + "fft"] = fft
  del fft
  print("Performed fft for", label, f"(took {round(time.time() - start, 2)} seconds)")

#do_fft(fakedata, "T")

do_fft(df_eeg_data, "FT7")

rng = np.arange(-fftsize / 2, fftsize / 2) * sampleduration / fftsize

def show_fft_data(frame_fft, frame_phase, signal=None, show=True, _start=0):
  graphs = [
    dcc.Graph(
        id="Graph1",
        figure={
            "data": [
                {"x": rng, "y": [np.abs(a) for a in frame_fft[1:]], "type": "line", "name": "magnitude"},
                {"x": rng, "y": [0] * fftsize, "type": "line", "name": "Baseline"},
            ],

            "layout": {
                "title": "FFT",
                "xaxis": {"title": "Frequency (Hz)"},
                "yaxis": {"title": "Magnitude"}
            }
        }
    ),
    dcc.Graph(
        id="Graph2",
        figure={
            "data": [
                {"x": rng, "y": [abs(a) for a in frame_phase[1:]], "type": "line", "name": "phase"},
                {"x": rng, "y": [0] * fftsize, "type": "line", "name": "Baseline"},
            ],

            "layout": {
                "title": "FFT Phase",
                "xaxis": {"title": "Phase"},
                "yaxis": {"title": "Frequency (Hz)"}
            }
        }
    ),
  ]

  if signal is not None:
    sigrng = np.arange(0, fftsize) / fftsize * sampleduration
    graphs.insert(0, dcc.Graph(
        id="Graph3",
        figure={
            "data": [
                {"x": sigrng, "y": signal[_start:_start+fftsize], "type": "line", "name": "phase"},
                {"x": sigrng, "y": [0] * fftsize, "type": "line", "name": "Baseline"},
            ],

            "layout": {
                "title": "Signal",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Amplitude"}
            }
        }
    ))
  
  if show:
    # Create app
    app = JupyterDash("Graphs")

    # Realtime bug workaround from 11 hours ago, hot off the presses
    # https://github.com/plotly/dash/issues/1907#issuecomment-1035931483
    del app.config._read_only["requests_pathname_prefix"]
    app.layout = html.Div(children=graphs)

    ###!!!! WARNING! The JupyterDash class is apparently very inefficient the first time it runs.
    ###!!!!          It may take up to ten seconds to load, during which the output will initially
    ###!!!!          appear to be blank. DO NOT FEAR! It will eventually be populated.
    app.run_server()
  
  return graphs

should_show_test_data = False #@param {"type": "boolean"}
#print(df_eeg_data.keys())
if should_show_test_data: show_fft_data(df_eeg_data["FT7fft"][0], df_eeg_data["FT7phase"][0], df_eeg_data["Time"])

# plt.plot(df_eeg_data[:500]["Time"], df_eeg_data[:500]["FT7"])
# plt.show()

# Normalization, correlation data used for EDA + Data Preprocessing
norm_df_eeg_data = (df_eeg_data - df_eeg_data.mean()) / df_eeg_data.std()
looking_for_indices = [int("you" in caption_text[int(i / samplerate)].split()) for i in range(len(norm_df_eeg_data))]

norm_df_eeg_data["Has_You"] = looking_for_indices
corr_matrix = norm_df_eeg_data.corr()

# Principle Component Analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

n = 2
pca = PCA(n_components=n)
#print(norm_df_eeg_data.columns[norm_df_eeg_data.isna().any()].tolist())
filtered_eeg_data = norm_df_eeg_data.drop(['FT7fft', 'FT7phase'], axis=1)
pc = pca.fit_transform(filtered_eeg_data)
pc_df = pd.DataFrame(data=pc, columns=[f'PC{i}' for i in range(n)])
var_df = pd.DataFrame({'var':pca.explained_variance_ratio_, 'PC':[f'PC{i}' for i in range(n)]})

"""# Initial EDA Dashboard
Shows Correlation Heatmap along with PCA = 2 scatterplot. Dropdown allows users to select which features they want to see correlations for.

TODO:
 - Set heatmap color scale to be static (-1, 1)
 - Improve padding between graphs
 - Add some per-feature graphs
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer

#import plotly.express as px
#px.data.tips()

from dash import Input, Output
import plotly.express as px
import math

srate = 512

def get_timestamp(i):
  timestamp = str(math.floor(i / srate / 3600) % 60).rjust(2, '0') + ":" + str(math.floor(i / srate / 60) % 60).rjust(2, '0') + ":"
  timestamp += str(math.floor(i / srate) % 60).rjust(2, '0')
  return timestamp

framewindow = 10
_res = None

sid = SentimentIntensityAnalyzer()
def update_metrics(slider):
  global _res
  i = max(0, slider * srate - 1) # int((time.time() - start) * srate) % len(df_eeg_data["FT7fft"])
  if i > len(df_eeg_data):
    return dash.no_update#, "Index " + str(i) + " out of bounds"
  word = caption_text[df_eeg_data["Caption"][i]]
  _words = word.strip().split()
  #print("A")
  scores = sid.polarity_scores(word)
  #print("B")
  timestamp = get_timestamp(i)
  sentiment = [
    html.H2("Sentiment Analysis"),
    html.Div(children=["Word: " + repr(word).replace("'", '"')]),
    html.Div(children=["Timestamp: " + timestamp]),
    html.Div(children=["Sentiment analysis: " + ", ".join("{0}: {1}".format(k, scores[k]) for k in sorted(scores.keys()))])
  ]
  vader_results = {word: sid.polarity_scores(word) for word in _words}
  graph_data = {"neu": [vader_results[word]["neu"] for word in _words],
                "word": _words}

  '''sentiment.append(dcc.Graph(
      figure=px.bar(
          graph_data, x="word", y="neu",
          color=["neu"]
      )
  ))'''
  
  #print("C")
  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html ?
  results = show_fft_data(df_eeg_data["FT7fft"][i], df_eeg_data["FT7phase"][i], df_eeg_data["Time"], show=False, _start=i)
  results = [
    html.Div(style={"display": "flex"}, children=results[:2])
  ] + results[2:]
  results = results + sentiment
  #print(results)
  _res = (time.time(), results)
  return results

from urllib.request import urlopen, Request
from io import BytesIO
from PIL import Image

user_agent = "MINTSRequester/1.0" #"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36"
bio = BytesIO(urlopen(Request("https://cdn.discordapp.com/attachments/911339282537529419/956940719648563250/EEG_10-10_system.png", headers={"User-Agent": user_agent})).read())

# eeg_img = np.array(Image.new("RGB", (500, 500), color=(255, 255, 255)))
eeg_img = np.array(Image.open(bio).convert('RGB'))

"""
Usaid's just-added stuff
df_eeg_data = normalized (norm_df_eeg_data)
df_raw_data = df_eeg_data (no normalization)
"""
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

external_sheets = ["https://fonts.googleapis.com/icon?family=Material+Icons"]

app = Dash(__name__, external_stylesheets=external_sheets)

@app.callback(
    Output(component_id='eda_corr', component_property='figure'),
    Input(component_id='eda_feats', component_property='value')
)
def update_corr_fig(input_value):
    cols_idx = [corr_matrix.columns.get_loc(c) for c in input_value]
    filtered_corr_mat = corr_matrix[input_value].iloc[cols_idx]
    corr_fig = px.imshow(filtered_corr_mat, color_continuous_scale='RdBu_r', title='Pearson-Coefficient Heatmap', range_color=[-1, 1])
    corr_fig.update_layout(transition_duration=500)
    return corr_fig

def display_time_series(series_feature):
    fig = go.Figure()
    for feat in series_feature:
        fig.add_trace(go.Scatter(x=df_eeg_data[feat].index, y=df_eeg_data[feat].values, name=feat))
    return fig


@app.callback(
    Output('time_series', 'figure'),
    Input('eda_feats', 'value')
)
def update_time_series(series_feature):
  return display_time_series(series_feature)

def eeg_heatmap(slider):
    pos = \
        {
            'FC2': [(288, 215), (315, 242)],
            'Iz': [(243, 499), (269, 526)],
            'T7': [(53, 262), (80, 288)],
            'C1': [(195, 261), (222, 288)],
            'C3': [(147, 261), (174, 288)],
            'F6': [(361, 159), (387, 186)],
            'CPz': [(243, 308), (269, 335)],
            'O1': [(184, 443), (211, 470)],
            'Fpz': [(243, 72), (269, 99)],
            'Ft10': [(469, 188), (496, 215)],
            'FC6': [(378, 209), (405, 236)],
            'CP5': [(107, 314), (134, 341)],
            'Fz': [(243, 166), (270, 193)],
            'AF4': [(301, 119), (328, 146)],
            'P9': [(44, 393), (71, 420)],
            'O2': [(301, 443), (328, 470)],
            'Cz': [(243, 262), (269, 288)],
            'CP6': [(378, 314), (405, 341)],
            'CP2': [(288, 309), (315, 336)],
            'P7': [(89, 374), (116, 401)],
            'T8': [(432, 262), (459, 288)],
            'P8': [(396, 374), (423, 401)],
            'FT9': [(16, 188), (43, 215)],
            'PO3': [(185, 407), (212, 434)],
            'P6': [(360, 365), (387, 392)],
            'F4': [(322, 163), (349, 190)],
            'Fp2': [(301, 82), (327, 109)],
            'FCz': [(243, 214), (269, 241)],
            'T9': [(5, 261), (32, 288)],
            'FC3': [(153, 212), (180, 239)],
            'C6': [(385, 262), (412, 288)],
            'AF8': [(355, 110), (382, 137)],
            'POz': [(243, 403), (270, 430)],
            'F3': [(163, 163), (190, 190)],
            'F10': [(441, 131), (468, 158)],
            'C4': [(338, 261), (365, 288)],
            'Pz': [(243, 357), (269, 384)],
            'P2': [(281, 358), (308, 385)],
            'FT10': [(469, 188), (495, 215)],
            'P5': [(125, 365), (152, 392)],
            'T10': [(480, 262), (507, 288)],
            'P1': [(204, 358), (231, 385)],
            'FC5': [(107, 209), (134, 236)],
            'P10': [(442, 393), (469, 420)],
            'CP3': [(152, 313), (179, 340)],
            'F2': [(281, 166), (308, 193)],
            'F1': [(204, 166), (231, 193)],
            'TP9': [(17, 335), (44, 362)],
            'P3': [(164, 360), (190, 387)],
            'FT7': [(63, 203), (90, 230)],
            'AF7': [(130, 110), (157, 137)],
            'PO4': [(300, 407), (327, 434)],
            'F5': [(124, 159), (151, 186)],
            'PO8': [(354, 416), (381, 443)],
            'TP7': [(63, 321), (90, 348)],
            'P4': [(322, 360), (348, 387)],
            'CP1': [(197, 309), (224, 336)],
            'F9': [(44, 131), (71, 158)],
            'AF3': [(184, 119), (211, 146)],
            'CP4': [(333, 313), (360, 340)],
            'Oz': [(243, 452), (269, 479)],
            'TP10': [(468, 335), (495, 362)],
            'C2': [(290, 261), (317, 288)],
            'F7': [(89, 150), (116, 177)],
            'AFz': [(243, 120), (269, 147)],
            'Fp1': [(185, 82), (211, 109)],
            'Nz': [(243, 24), (269, 51)],
            'PO7': [(131, 416), (158, 443)],
            'F8': [(396, 150), (423, 177)],
            'FT8': [(422, 203), (449, 230)],
            'FC1': [(197, 215), (224, 242)],
            'C5': [(100, 262), (127, 288)],
            'TP8': [(422, 321), (449, 348)],
            'FC4': [(332, 212), (359, 239)],
        }
    fig = px.imshow(eeg_img)
    for feat in df_eeg_data.columns:
        if feat in pos:
            color = 'red'
            weight = abs((df_eeg_data[feat].max() - df_eeg_data.iloc[slider*srate][feat]) / (df_eeg_data[feat].max() - df_eeg_data[feat].min()))
            if df_eeg_data.iloc[slider*srate][feat] < 0:
                color = 'blue'
            fig.add_shape(editable=False, fillcolor=color, type='circle', \
                x0=pos[feat][0][0], y0=pos[feat][0][1], x1=pos[feat][1][0], y1=pos[feat][1][1], \
                    opacity=weight)
    return fig

@app.callback(
    Output('eeg_heatmap', 'figure'),
    Input('time_slider', 'value')
)
def eeg_heatmap_callback(slider):
  return eeg_heatmap(slider)

enable_automatic_play = False
@app.callback(
    Output('other_charts', 'children'),
    Input('time_slider', 'value')
)
def metrics(slider):
  global _res

  _res = "called"
  try:
    return update_metrics(slider)
  except Exception as e:
    _res = e

@app.callback(
    Output("play_button", "children"),
    Input("play_button", "n_clicks")
)
def pbtn(n_clicks):
  global enable_automatic_play
  if n_clicks is None:
    return dash.no_update
  
  if enable_automatic_play:
    enable_automatic_play = False
    return ["play_arrow"]
  else:
    enable_automatic_play = True
    return ["pause"]

minTime = df_eeg_data['Time'].min()
maxTime = df_eeg_data['Time'].max()

@app.callback(
    Output("time_slider", "value"),
    State("time_slider", "value"),
    Input('play_interval', 'n_intervals')
)
def updt(slider, intervals):
  ctx = dash.callback_context
  if not ctx.triggered or not enable_automatic_play: return dash.no_update

  return max(minTime, min(maxTime - 1, slider + 1))

@app.callback(
    Output("timestamp", "children"),
    State("time_slider", "value"),
    Input('play_interval', 'n_intervals')
)
def ts_updt(slider, interval):
  return [str(get_timestamp(slider))]

@app.callback(
    Output("eda_feats", "value"),
    Input("show_all_button", "n_clicks")
)
def proc(n_clicks):
  if n_clicks is None or n_clicks == 0: return dash.no_update#, ""
  return corr_matrix.columns

pca_fig = px.scatter(pc_df, x='PC0', y='PC1', title='PCA Scatterplot')

yt_script = """
((style) => {
    window.videos = window.videos || {};
    const videos = [vidids];
    
    function onStateChange(event) {
        
    }
    
    window.onYouTubeIframeAPIReady = function() {
        Object.keys(videos).forEach(video) {
            videoData = videos[video];
            let player = new YT.Player(video, {
                width:  "260",
                height: "960",
                videoId: "",
                playerVars: {
                    playsinline: 1
                },
                events: {
                    onReady: (event) {
                        console.log("Player ready");
                    },
                    onStateChange: onStateChange
                }
            });
            
            window.videos[video] = player;
        }
    };
    
    if(!window.ytLoaded) {
        window.ytLoaded = true;
        
        var tag = document.createElement("script");
        tag.src = "https://www.youtube.com/iframe_api";
        var firstScript = document.getElementsByTagName("script")[0];
        console.log(firstScript);
        firstScript.parentNode.insertBefore(tag, firstScript);
    }
    
    console.log("Script inserted");
    return style;
})
"""

videoValues = {
    'eyestream_video': {
        'id': '2HBYjxtLcHY',
        'width': '0',
        'height': '0'
    },
    
    'main_video': {
        'id': 'HbVf7pmogVI',
        'width': '0',
        'height': '0'
    }
}

app.clientside_callback(
    yt_script.replace("[vidids]", ),
    Output("hidden-div", "style"),
    Input("hidden-div", "style")
)

app.layout = html.Div(children=[
    html.P(id="hidden-div", style={"display": "none"}, title="none"),
    html.H1(children='MINTS Biometric Analysis'),
    html.Div(children=[
        html.H2(children='Exploratory Data Analysis'),
        html.H4(children='Data Features'),
        html.Button("Show All Features", id="show_all_button", n_clicks=0),
        dcc.Dropdown(id='eda_feats', options=corr_matrix.columns, value=corr_matrix.columns[:10], multi=True),
        html.Div(children=[
            html.H3(children='Correlations'),
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(id='eda_corr')
                ], style={'display': 'inline-block', 'width': '30%', 'verticalAlign': 'top'}),
                html.Div(children=[
                    dcc.Graph(id='time_series', figure=display_time_series(corr_matrix.columns[:10])),
                ], style={'display': 'inline-block', 'width': '70%', 'verticalAlign': 'top'}),
            ]),
        ]),
        html.Div(children=[
            html.Div(children=[
                html.H3(children="Eyestream"),
                html.Div(id="eyestream_vid")
            ]),
        ], style={"display": "flex"}),
        html.H2(children='Per-timestep Analysis'),
        html.Div(children=[
          html.H4(children='Timestep', style={"backgroundColor": "inherit"}),
          html.Div(children=[
              html.Div(children=[
                html.Button("play_arrow", id="play_button", className="material-icons"),
                html.P(children=["00:00:00"], id="timestamp", style={"backgroundColor": "inherit"})
              ], style={"float": "left", "width": "3%"}),
              html.Div(children=[dcc.Slider(id='time_slider', value=minTime, min=minTime, max=maxTime)], style={"float": "right", "width": "96%"})
          ])
        ], style={"position": "sticky", "top": "10px", "backgroundColor": "rgba(255, 255, 255, 0.4)", "zIndex": "10000"}),
        html.H4('EEG Heatmap'),
        dcc.Graph(id='eeg_heatmap', figure=eeg_heatmap(minTime)),
        html.Div(children=[
            html.H4(children="Textual Charts"),
            html.Div(id="other_charts", children=update_metrics(minTime))
        ]),
        html.Div(children=[
            html.H4(children='Principle Component Analysis'),
            dcc.Graph(id='eda_pca', figure=pca_fig)
        ])
    ]),
    dcc.Interval(id='play_interval', interval=1000, n_intervals=0)
])

app.run_server() #mode="inline", debug=True, port=1051

exit()
import tensorflow as tf

from tqdm import tqdm

fftwindow = 512
labels = ("FT7", "FT8", "TP7", "TP10")
looking_for = "you"
to_range = len(norm_df_eeg_data) - fftwindow

looking_for_indices = [i for i in range(to_range) if (looking_for in caption_text[int(i / samplerate)].split())]

df_eeg_found = norm_df_eeg_data[norm_df_eeg_data.index.isin(looking_for_indices)]
df_eeg_not_found = norm_df_eeg_data[~norm_df_eeg_data.index.isin(looking_for_indices)]

df_eeg_found.reset_index(inplace=True)
df_eeg_not_found.reset_index(inplace=True)
None

#df_eeg_found.to_csv('eeg_data_found.csv')
#df_eeg_not_found.to_csv('eeg_data_not_found.csv')

def convert_inputs1(data):
  tensors = []

  for i in tqdm(range(len(data))):
    tensor = [data[label][i] for label in labels]
    tensors.append(tensor)
  
  return tensors

base_dataset_found = tf.data.Dataset.from_tensor_slices(convert_inputs1(df_eeg_found))
base_dataset_not_found = tf.data.Dataset.from_tensor_slices(convert_inputs1(df_eeg_not_found))

LSTM_BATCH_LEN = 16 #@param
BATCH_SIZE = 16 #@param

def transform_data(fft_data):
  return [[tf.abs(fft_data)] * LSTM_BATCH_LEN]

def convert_x_tensors(data):
  return [tf.convert_to_tensor(np.abs(row), dtype=tf.float64) for row in data["FT7fft"]]

x_tensors_found = convert_x_tensors(df_eeg_found)
x_tensors_not_found = convert_x_tensors(df_eeg_not_found)

fft_dataset_found = tf.data.Dataset.from_tensor_slices(x_tensors_found).map(transform_data)
fft_dataset_not_found = tf.data.Dataset.from_tensor_slices(x_tensors_not_found).map(transform_data)

#def gen_y_dataset(data):
#  return [[int(looking_for in caption_text[int(i / samplerate)].split())] for i in range(len(data))]

#y_dataset_found = tf.data.Dataset.from_tensor_slices(gen_y_dataset(df_eeg_found))
#y_dataset_not_found = tf.data.Dataset.from_tensor_slices(gen_y_dataset(df_eeg_not_found))

y_dataset_found = tf.data.Dataset.from_tensor_slices([[1]]).repeat(len(df_eeg_found))
y_dataset_not_found = tf.data.Dataset.from_tensor_slices([[0]]).repeat(len(df_eeg_not_found))

x_dataset_found = tf.data.Dataset.zip((base_dataset_found, fft_dataset_found))
x_dataset_not_found = tf.data.Dataset.zip((base_dataset_not_found, fft_dataset_not_found))

dataset_found = tf.data.Dataset.zip((x_dataset_found, y_dataset_found))
dataset_not_found = tf.data.Dataset.zip((x_dataset_not_found, y_dataset_not_found))

# Split into test and train
train_percent = 0.8
dataset_found_test = dataset_found.skip(int(train_percent * len(dataset_found)))
dataset_not_found_test = dataset_not_found.skip(int(train_percent * len(dataset_not_found)))
dataset_found = dataset_found.take(int(train_percent * len(dataset_found)))
dataset_not_found = dataset_not_found.take(int(train_percent * len(dataset_not_found)))

joined_dataset = tf.data.Dataset.sample_from_datasets([dataset_found.repeat(), dataset_not_found.repeat()])
joined_dataset_test = dataset_found_test.concatenate(dataset_not_found_test)

dataset = joined_dataset.batch(BATCH_SIZE)
dataset_test = joined_dataset_test.shuffle(len(joined_dataset_test)).batch(BATCH_SIZE)

input1 = tf.keras.layers.Input(shape=(len(labels),))
input2 = tf.keras.layers.Input(shape=(LSTM_BATCH_LEN, fftsize))

x1 = tf.keras.layers.Dense(48, activation="sigmoid")(input1)
x2 = tf.keras.layers.LSTM(256)(input2)
x2 = tf.keras.layers.Dense(48, activation="sigmoid")(x2)
x = tf.keras.layers.Dense(64)(tf.concat([x1, x2], axis=-1))
x = tf.keras.layers.Dense(16)(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=[input1, input2], outputs=x)

tf.keras.utils.plot_model(model, show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True, dpi=96, show_layer_activations=True)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=["accuracy"])

history = model.fit(dataset, epochs=5, steps_per_epoch=len(df_eeg_data), validation_data=dataset_test)

#!while 1 do; sleep 1; nvidia-smi; done;

## PROBLEM! The data is very uneven (i.e., the "you"s are quite sparse)
## TODO: Ask Usaid + others about evening out the "you"s vs the not "you"s so
##       that we don't have this issue where prediction is unfairly biased
##       towards not "you" regardless of the inputs.

items = list(iter(dataset.take(8)))

items[-1][-1]

# Usaid: Exploratory Data Analysis Stuff I'm trying out
import sys
#!{sys.executable} -m pip install -U pandas-profiling[notebook]
#!jupyter nbextension enable --py widgetsnbextension

pd.__version__

from pandas_profiling import ProfileReport
profile = ProfileReport(df_eeg_data, minimal=True)
profile.to_file('eda.html')

df_eeg_data.to_csv('eeg_data.csv')

"""# MFCCs, extracting features from audio
The current state-of-the-art feature extraction used for speech and speaker recognition is Mel-Frequency Cepstral Coefficients (MFCCs). These were first conceptualized and used in the 80s, and more people continue to find applications where the MFCCs have been helpful and perform quite well. This paper from Sato and Obuchi https://www.jstage.jst.go.jp/article/imt/2/3/2_3_835/_pdf (2007) uses a very simple implementation of MFCCs to get 66.4% accuracy in distinguishing between the emotions (hot anger, neutral, sadness, and happiness). When just looking at the emotions (hot anger, and neutral) the simple algorithm gets 98.75% accuracy. Both of these improve on previous attempts that used Prosodic feataures (pitch, loudness, tempo, rhythm). This shows that MFCCs are a good feature for us to extract if we are looking to gather some emotive responses from the audio to pair with the biometric information. 

\\
While this might be more than enough to go off of, if we are looking to take it a step further, the paper "Recognition of Human Speech Emotion Using Variants of Mel-Frequency Cepstral Coefficients" published in 2018 can help provide some direction for which variations to use. 
https://www.researchgate.net/profile/Lenin-Nc/publication/321755687_Linear_Synchronous_Reluctance_Motor-A_Comprehensive_Review/links/5b18bdd90f7e9b68b424b63e/Linear-Synchronous-Reluctance-Motor-A-Comprehensive-Review.pdf#page=490
## Visualizing Mel Spectrograms
"""

import librosa
import librosa.display

# Load the yt_wav file into a signal matrix

signal, audio_sample_rate = librosa.load(yt_wav)

# get the Mel filter banks. key to getting the spectrogram 
# extract vanilla spectrogram, apply mel filter banks, get the mel spectrogram
# answers the question what is the weight that should be applied to each frequency? i.e. where are the important mel bands

filter_banks = librosa.filters.mel(n_fft=4096, sr=audio_sample_rate, n_mels=15)
plt.figure(figsize=(25, 10))
librosa.display.specshow(filter_banks, sr=audio_sample_rate, x_axis="linear")
plt.colorbar()
plt.show()

# the higher weights indicate where the center of the mel bands are, you can see the 15 mel bands as blocks, and the intensity shows how high the mel peak is

mel_spectrogram = librosa.feature.melspectrogram(signal, sr=audio_sample_rate, n_fft=4096, hop_length=512, n_mels=15)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=audio_sample_rate)
plt.colorbar()
plt.show()

"""# Extracting MFCCs from the yt_wav"""

# extract the 15 highest MFCCs (arbitrary 15)

mfccs = librosa.feature.mfcc(signal, n_mfcc=15, sr=audio_sample_rate)
mfccs.shape

# visualize the MFCCs, spectrum of a spectrum
plt.figure(figsize=(25, 10))
librosa.display.specshow(mfccs, x_axis="time", sr=audio_sample_rate)
plt.colorbar()
plt.show()

# calculate delta and delta2 MFCCs (first and second derivatives)
delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs, order=2)

# visualize the derivatives of the MFCCs
plt.figure(figsize=(25, 10))
librosa.display.specshow(delta_mfccs, x_axis="time", sr=audio_sample_rate)
plt.colorbar()
plt.show()

plt.figure(figsize=(25, 10))
librosa.display.specshow(delta2_mfccs, x_axis="time", sr=audio_sample_rate)
plt.colorbar()
plt.show()

combined_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
combined_mfccs.shape