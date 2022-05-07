# Visualization

## Installation

Packages can be installed with the following command:

`python -m pip install tqdm pandas scipy spacy dash jupyter-dash pandas==1.2.0 mne[data] gdown yt-dlp webvtt-py librosa Pillow matplotlib nltk detoxify`

## MINTS Data Installation
There are two options with regards to the MINTS experiment data. You can either
extract the proper MINTS.zip file into `visualization/nlp_data`,
or you can provide a URL and password to have the system automatically download
and extract a file (like `python visualization.py <url_to_mints.zip> <password_to_open_mints.zip>`).
Please note that, now that this file is more easily available publicly, I've
taken down the MINTS.zip file hosted on my personal server.

## First-time processing
The first time the program runs, it will probably download additional models and run them
on the experiment for visualization purposes. Depending on the specs of your computer, this
may take up to an hour. Don't worry; the results will be cached so that restarting the
dashboard should take just a few seconds each time.

## Contents
 - Visualization.py
   - Our Python dashboard program.

## Running
Simply run `python3 visualization.py` to start the main visualization script.

## Link to Online Environment
To run this visualization tool without installing it on your local machine, please use
[this link](https://colab.research.google.com/drive/1nGuMVrvAVbVV_ODxyO1mOdgAR7FHm-DZ?usp=sharing) to access the hosted
version on Google Colab. Please note that this version is much older than the current `.py` file.
