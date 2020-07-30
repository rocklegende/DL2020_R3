# DL2020_R3

Spotify song recommendation system, which recommends similar songs based on a given input song.
The dataset that is used is the publicly available dataset from Kaggle, 
called [the Spotify Dataset](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks), with 160k+ Tracks released inbetween 1921 and 2020.

## How to execute

### Install required python modules
Please make sure you have the required python modules in ```requirements.txt``` installed.

### Download data
Please download the CSV dataset from https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks

Create new folder ```data``` inside ```src```

```buildoutcfg
cd src
mkdir data
```

Put all files from the dataset into ```src/data/``` (So the full path for data.csv is ```src/data/data.csv```)

### Training
First execute ```src/train.py``` to train the model. After this process the 
model parameters are stored in ```src/model.pt```

### Get recommendations for Spotify song id
Execute ```test.py``` to get song recommendations. If you execute
this Python script you will be asked to input a Spotify song id. Please use Kaggle
to search for the song id of the song you want to input. For this go to ```data.csv``` file
and just use the columns ```artists```, ```id```, ```popularity``` and ```name```.

Please set the filter on ```popularity``` from 20 to 100.
Use the filter for ```artists``` or ```name``` to search for artists or track names.

After you found a track, copy the ```id``` and paste it into the prompt.

Output are topk recommendations for the input song.

### Default Example (no input)
If you just want to get a default example and not input a song id, then execute 

```default_example.py``` 

This will show recommendations based on the first song of the dataset.