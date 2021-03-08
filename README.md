#Eluvio challenge- Scene Segmentation

Objective is to predict the probability that a shot boundary is a scene boundary.

For each movie, we use the mean of pairwise distances of given features. The shots of different scenes should have higher distance values when compared to shots of same scenes. We use this logic to predict the probabilities.

##Implementation
Have the 64 pickle files data in a folder ('data-dir') and run the following command. ('output-dir' denotes where you want the output files to be stored)

`python main.py data-dir output-dir`

##Validation
To evaluate the files run:

`python elv-ml-challenge/evaluate_sceneseg.py output-dir`

##Results
```
(py) C:\Users\nagas\PycharmProjects\eluvio_challenge>python elv-ml-challenge\evaluate_sceneseg.py output
# of IMDB IDs: 64
Scores: {
    "AP": 0.43432948488202827,
    "mAP": 0.456352123550865,
    "Miou": 0.46784292015853995,
    "Precision": 0.5318621161859483,
    "Recall": 0.35784778487868607,
    "F1": 0.4101209096486932
}
```

```
(py) C:\Users\nagas\PycharmProjects\eluvio_challenge>python elv-ml-challenge\evaluate_sceneseg.py data
# of IMDB IDs: 64
Scores: {
    "AP": 0.4418872028438688,
    "mAP": 0.45644015956781614,
    "Miou": 0.4541480053002174,
    "Precision": 0.2761656092479825,
    "Recall": 0.7473442326299846,
    "F1": 0.39309552999275693
}
```