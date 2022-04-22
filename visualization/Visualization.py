# -*- coding: utf-8 -*-
"""
Original file is located at https://colab.research.google.com/drive/1nGuMVrvAVbVV_ODxyO1mOdgAR7FHm-DZ
This version has been altered heavily for use on a local machine.
"""
import argparse
import base64
import json

ap = argparse.ArgumentParser(description="Visualize EEG data")

ap.add_argument("zip_password", help="The password for the MINTS zip (not included for security reasons)")

args = ap.parse_args()


# Prepare environment
import os
import subprocess
import re

# Kept so you know what to install
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
# Do not enable interactive mode
#plt.ion()

from urllib.request import urlretrieve

# Import dash selectively
import dash

try:
    # Test if we're in a a Colab environment
    import google.colab 
    from jupyter_dash import JupyterDash
    class Dash(JupyterDash):
        def run_server(self, *args, **kwargs):
            return JupyterDash.run_server(self, *args, mode="inline", debug=True, **kwargs)
except ImportError:
    from dash import Dash

to_open = "MINTS.zip" # Replace with LargeMINTS.zip for the large dataset
nlp_data = "large_nlp_data" if "Large" in to_open else "nlp_data"

if not os.path.exists(os.path.join(nlp_data, "./2022_01_14_T04_U002_EEG01/2022_01_14_T04_U002_EEG01.vhdr")):
    zip_password = args.zip_password

    root_dir = os.path.split(os.path.abspath("."))[0]
    zip_path = os.path.join(root_dir, "data", to_open)
    os.makedirs(nlp_data, exist_ok=True)

    print("Attempting to call 7zip")
    subprocess.check_call(["7z", "-y", "x", "-p" + zip_password, zip_path], cwd=os.path.abspath(nlp_data))

forest_predictions_path = os.path.join(nlp_data, "forest_toxicity_predictions.csv")
if not os.path.exists(forest_predictions_path):
    print("Downloading", forest_predictions_path)
    urlretrieve("https://personal.utdallas.edu/~rdc180001/forest_toxicity_predictions.csv", forest_predictions_path)

forest_predictions = pd.read_csv(forest_predictions_path, header=None)

forest_predictions_pickle = os.path.join(nlp_data, "forest_regressor.pickle")
if not os.path.exists(forest_predictions_pickle):
    print("Downloading", forest_predictions_pickle)
    urlretrieve("https://personal.utdallas.edu/~rdc180001/new_regressor.pickle", forest_predictions_pickle)

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

# Normalization, correlation data used for EDA + Data Preprocessing
norm_df_eeg_data = (df_eeg_data - df_eeg_data.mean()) / df_eeg_data.std()
looking_for_indices = [int("you" in caption_text[int(i / samplerate)].split()) for i in range(len(norm_df_eeg_data))]

norm_df_eeg_data["Has_You"] = looking_for_indices
corr_matrix = norm_df_eeg_data.corr()

# Principle Component Analysis
from sklearn.decomposition import PCA

n = 2
pca = PCA(n_components=n)

try:
    filtered_eeg_data = norm_df_eeg_data.drop(['FT7fft', 'FT7phase'], axis=1)
except KeyError:
    filtered_eeg_data = norm_df_eeg_data
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

srate = samplerate #512

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

def _update_corr_fig(input_value):
    cols_idx = [corr_matrix.columns.get_loc(c) for c in input_value]
    filtered_corr_mat = corr_matrix[input_value].iloc[cols_idx]
    corr_fig = px.imshow(filtered_corr_mat, color_continuous_scale='RdBu_r', title='Pearson-Coefficient Heatmap', range_color=[-1, 1])
    corr_fig.update_layout(transition_duration=500)
    return corr_fig

@app.callback(
    Output(component_id='eda_corr', component_property='figure'),
    Input(component_id='eda_feats', component_property='value')
)
def update_corr_fig(input_value):
    return _update_corr_fig(input_value)

range_in_seconds = (np.arange(len(forest_predictions[0])) / samplerate)

def display_time_series(series_feature, title="Time Series Data", max_features=20):
    fig = go.Figure(layout={"title": title})
    for i, feat in enumerate(series_feature):
        try:
            fig.add_trace(go.Scatter(x=range_in_seconds, y=df_eeg_data[feat].values, name=feat))
        except KeyError:
            pass # Quietly ignore
        
        if i + 1 > max_features:
            fig.update_layout({"title": title + f"(only first {max_features} shown for performance reasons)"})
            break
    return fig

forest_figure = go.Figure(layout={"title": "Random Forest Predictors"})
forest_figure.add_trace(go.Scatter(x=range_in_seconds, y=forest_predictions[0], name="Toxicity"))

@app.callback(
    Output('time_series', 'figure'),
    Input('viz_feats', 'value')
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
                    opacity=weight, name=feat)
    # legend
    fig.update_layout(showlegend=False, title="EEG Heatmap")

    # x axis
    fig.update_xaxes(visible=False)

    # y axis
    fig.update_yaxes(visible=False)
    return fig

@app.callback(
    Output('eeg_heatmap', 'figure'),
    Input('eeg_slider', 'value')
)
def eeg_heatmap_callback(slider):
  return eeg_heatmap(slider)

enable_automatic_play = False
'''@app.callback(
    Output('other_charts', 'children'),
    Input('time_slider', 'value')
)
def metrics(slider):
  global _res

  _res = "called"
  try:
    return update_metrics(slider)
  except Exception as e:
    _res = e'''

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

'''@app.callback(
    Output("time_slider", "value"),
    State("time_slider", "value"),
    Input('play_interval', 'n_intervals')
)
def updt(slider, intervals):
  ctx = dash.callback_context
  if not ctx.triggered or not enable_automatic_play: return dash.no_update

  return max(minTime, min(maxTime - 1, slider + 1))'''

'''@app.callback(
    Output("timestamp", "children"),
    State("time_slider", "value"),
    Input('play_interval', 'n_intervals')
)
def ts_updt(slider, interval):
  return [str(get_timestamp(slider))]'''

@app.callback(
    Output("eda_feats", "value"),
    Input("show_all_button", "n_clicks")
)
def proc(n_clicks):
  if n_clicks is None or n_clicks == 0: return dash.no_update#, ""
  return corr_matrix.columns

pca_fig = px.scatter(pc_df, x='PC0', y='PC1', title='Principle Component Analysis Scatterplot')

yt_script = open("yt_script.js", "r", encoding="utf-8").read()

videoValues = {
    'eyestream_video': {
        'id': '9zMJs_z-3Y8',
        'width': '260',
        'height': '960'
    },
    
    'main_video': {
        'id': '8v557rZ08SI',
        'width': '1550', # 960x540 works well, 1440x810
        'height': '960'
    }
}

app.clientside_callback(
    yt_script.replace("[vidids]", json.dumps(videoValues)),
    Output("hidden-div", "style"),
    Input("hidden-div", "style")
)

initial_feature_selection = corr_matrix.columns[:10]

'''app.clientside_callback(
    """function() {
        $('cor_update').css('filter', 'gray');
        return [];
    }""",
    Output("hidden-div", "children"),
    Input("show_all_button", "n_clicks"),
    Input("eda_feats", "n_clicks")
)'''

app.layout = html.Div(children=[
    html.P(id="hidden-div", style={"display": "none"}, title="none"),
    html.H1(children='MINTS Biometric Analysis'),
    html.Div(children=[
        html.Div(children=[
            html.Center(children=[
                html.H2(children="Dataset Visualization"),
                ]),
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(id='time_series', figure=display_time_series(initial_feature_selection)),
                    html.Button("Show All Features (may cause lag)", id="show_all_button1", n_clicks=0),
                    dcc.Dropdown(id='viz_feats', options=corr_matrix.columns, value=initial_feature_selection, multi=True),
                ], style={'display': 'inline-block', 'width': '49%',
                          'height': '100%', 'border-right': '', 'padding': '0', 'margin': '0', 'margin-bottom': '5px'}),
                html.Div(children=[
                    dcc.Graph(id="eeg_heatmap", figure=eeg_heatmap(minTime)),
                    html.Div(style={"height": "34px", "display": "inline-block"}),
                    dcc.Slider(id='eeg_slider', value=minTime, min=minTime, max=maxTime)
                    # ], style={"float": "right", "width": "96%"})
                ], style={'display': 'inline-block', 'width': '49%',  # 'min-height': '500px',
                          'height': '100%', 'padding': '0', 'margin': '0', 'margin-bottom': '5px'})
            ], style={'width': '100%'}),
            html.Div(children=[
                html.H2(children="Videos"),
                html.Div(children=[
                    html.Div(children=[
                        html.Center(children=[html.H3(children="Eyestream")]),
                        html.Div(id="eyestream_video"),
                    ], style={'display': 'inline-block', 'margin': '5px', 'border': '1px dotted gray'}),
                    html.Div(children=[
                        html.Center(children=[html.H3(children="Main Video")]),
                        html.Div(id="main_video")
                    ], style={'display': 'inline-block', 'margin': '5px', 'border': '1px dotted gray'})
                ], style={"display": "flex", 'border': '1px solid gray'}),
            ])
        ], style={'border': '1px solid black', 'width': '99%'}),
    ]),
    html.Div(children=[
        html.H2(children='Exploratory Data Analysis'),
        html.Div(children=[
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(id='eda_corr', figure=_update_corr_fig(initial_feature_selection),
                              config={"editable": False, "staticPlot": True}),
                    html.Div(children=[
                        html.Button("Show All Features", id="show_all_button", n_clicks=0),
                        dcc.Dropdown(id='eda_feats', options=corr_matrix.columns, value=initial_feature_selection,
                                     multi=True),
                    ], style={'display': 'inline-block', 'margin-left': '30px', 'margin-right': '5px',
                              'margin-bottom': '5px'}),
                ], style={'display': 'inline-block', 'width': '100%'}),
                html.Div(children=[
                    #html.H4(children='Principle Component Analysis'),
                    dcc.Graph(id='eda_pca', figure=pca_fig,
                              config={"editable": False, "staticPlot": True})
                ], style={})
            ]),
        ])
    ], style={'border': '1px solid black'}),
    html.Div(children=[
        html.H2(children='Models'),
        html.Div(children=[
            html.H2(children="Sentiment Analysis"),
            dcc.Graph(id='video_sentiment', figure=forest_figure)
        ], style={"border": "1px solid gray"})
    ], style={"border": "1px solid black"})
])

app.run_server() #mode="inline", debug=True, port=1051

exit()

# Usaid: Exploratory Data Analysis Stuff I'm trying out
import sys
#!{sys.executable} -m pip install -U pandas-profiling[notebook]
#!jupyter nbextension enable --py widgetsnbextension

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