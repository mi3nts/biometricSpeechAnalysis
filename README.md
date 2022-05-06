# MINTS Biometric Speech Analysis Project


## Toxicity-Biometric Relationship

We used Detoxify to generate the toxicity of the words being spoken based solely on the captions alone. The libraries were quite slow in generating a toxicity score for each caption, so memoization techniques were used in generating the toxicity scores for each timestep. To further increase iteration cycles, python pickles were used to avoid unnecessary recomputation across a larger time scale. The EEG data was normalized to prevent any one feature from dominating in terms of sheer scale of the units. It was also denoised by applying a Fourier Transform to find the top 5 principle frequencies of each node and then using an Inverse Fourier Transform to generate the pure wave. Then, using the normalized and denoised biometric data, we used SciKit Learn to fit a few different models and coarsely tuned some hyperparameters using the test set. The final model was a Random Forest Regressor of 100 trees created with an 80/20 training-test split. The regressor was trained to predict the toxicity based solely on the biometrics data. To use the regressor in the dashboard and in other places, it was saved and stored in a MINTS git repository as well as hosted through the UT Dallas website.

The work can be found in the [toxicity_prediction](toxicity_prediction/README.md) directory

## EEG-Assisted Transcription

We used Pytorch to combine a pretrained Wav2Vec2 model and a custom EEG model to perform speech transcription. Wav2Vec2 extracts important audio features which are combined with EEG features to be fed into the model. The model applies a 1D convolutional layer to the EEG data then uses a fully-connected network to output per-character probabilities. Similar to the original Wav2Vec2, the model outputs a set 40-character sequence, which requires a logit per-character per-sequence slot. Final output text is chosen using the argmax of the logits output layer.
The model training pipeline was created using Pytorch. Initially, when the video audio data is read, it is upsampled to the 16000Hz (Wav2Vec2’s input frequency) in order for Wav2Vec2 to extract features from it. Text captured from YouTube’s automatic captioning system is the intended target in the dataset. The text is attributed to the different rows in the input dataset based on what text chunk from the captioning system contains the most overlap with the batch in an input minibatch.

The work can be found in the [transcription_model](transcription_model/README.md) directory

## Dashboard

The dashboard was built using pydash and Plotly and a number of data analytic Python modules (pandas, numpy, etc.). We wanted it to be intuitive, and so we leveraged several pydash features to ensure that the user has configurable control over the data displayed. Several quality-of-life improvements were also implemented, including visualization and synchronizing of the video streams from the biometric eye cam/frontal cam.
Most of the data included from the models is imported from static files either describing their training and evaluation systems or containing the raw model data loaded into the dashboard for live evaluation. The videos are hosted as unlisted YouTube videos that we synchronized manually using JavaScript with the rest of the project. They appear on the dashboard using YouTube’s Iframe API.

The work can be found in the [visualization](visualization/README.md) directory
