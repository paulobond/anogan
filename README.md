This repo forks of https://github.com/LeeDoYup/AnoGAN

To train the model:

1/ create a folder called `data`. Under that folder, create a subfolder with the name 
of your dataset and fill it with normal images.
You can generate a dataset called "moons" by running `generate_dataset_moon.py`

2/ Run `main.py` by setting a flag `train = True`. If a model has already been trained
with the same dataset name and params (batchsize and image size), then the session is saved 
in a folder called `checkpoint`. The session will automatically be loaded.

To run the anormality detection:

1/ create a folder called `test_data`. Under that folder, create a subfolder with the name 
of your dataset and fill it with anormal or normal images.
You can generate a dataset of anormal "moons" with `generate_dataset_moon.py`

2/ Run `main.py` by setting a flag `test = True`. If `train = False`, the model should have
 already been trained and a session should be saved 
in a folder called `checkpoint`. The session will automatically be loaded. 
