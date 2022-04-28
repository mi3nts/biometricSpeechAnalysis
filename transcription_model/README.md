Use `python initial_model.py` to run the model.

This was made using Pytorch Lightning (which uses Pytorch on the backend while providing useful training routines/functionality such as a training dashboard and specific sampling methods).

How the code is set up:
 - `initial_model.py` = contains the main model training routine
 - `model_utils.py` = contains the associated Pytorch/Pytorch Lightning modules that make up the model/dataset/optimizer
 - `read_data.py` = copy of `../read_data.py` from root, could be easily removed with a few modifications to how other scripts access `read_data.py`'s functions. Provides utility functions to read the EEG/YT data. 
