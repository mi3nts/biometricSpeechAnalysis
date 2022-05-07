# Visualization Dashboard

## Introduction

This package (`visualization`) provides an interactive dashboard that enables data exploration and
examination of recorded runs of two machine learning models. Most of this data is downloaded in a
static form from hard-coded URLs that can be swapped as necessary.

## Contents
 - Visualization.py
   - Our Python dashboard program.
 - VisualizationSnapshotMarch26th.ipynb
   - An early version of our dashboard code as an interactive Jupyter notebook.

## Prerequisites

The packages necessary to run the dashboard can be installed with the following command:

`python -m pip install tqdm pandas scipy spacy dash jupyter-dash pandas==1.2.0 mne[data] gdown yt-dlp webvtt-py librosa Pillow matplotlib nltk detoxify`

You may want to do this in a `virtualenv`, as this can get quite messy.

## MINTS Data Installation
There are two options with regards to the MINTS experiment data. You can either
extract the proper MINTS.zip file (which contains experimental data in a specified
format) into `visualization/nlp_data`, or you can provide a URL and password to have the
system automatically download and extract an appropriate file (like
`python visualization.py <url_to_mints.zip> <password_to_open_mints.zip>`).
Please note that, now that this file is more easily available publicly, I've
taken down the MINTS.zip file hosted on my personal server, so the exact URLs found
in the Jupyter notebook will not work.

Please note that if you run `visualization.py` without arguments and without a valid
`nlp_data` directory, it will assume that you have specified a URL and password for
a `MINTS.zip` file to download.

## Running the Dashboard
Simply run `python3 visualization.py` to start the main visualization script. As soon as
the program finishes loading the data, it will display a localhost URL in the console that
you can use to connect to the dashboard's frontend.

### First-time processing
The first time the program runs, it will probably download additional models and run them
on the experiment for visualization purposes. Depending on the specs of your computer, this
may take up to an hour. Don't worry; the results will be cached so that restarting the
dashboard should take just a few seconds each time.


## Link to Online Environment
To run this visualization tool without installing it on your local machine, please use
[this link](https://colab.research.google.com/drive/1nGuMVrvAVbVV_ODxyO1mOdgAR7FHm-DZ?usp=sharing) to access the hosted
version on Google Colab. Please note that this version is much older than the current `.py` file (and, as before,
any hardcoded URLs it used to download `MINTS.zip` or similar have been removed for security's sake).
