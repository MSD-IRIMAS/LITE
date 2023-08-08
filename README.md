# LITE: Light Inception with boosTing tEchniques for Time Series Classification

This is the source code of our paper "[LITE: Light Inception with boosTing tEchniques for Time Series Classification](https://germain-forestier.info/publis/dsaa2023.pdf)" accepted at the 10th IEEE International Conference on Data Science and Advanced Analytics ([DSAA 2023](https://conferences.sigappfr.org/dsaa2023/)) in the  Learning from Temporal Data ([LearnTeD](https://dsaa2023.inesctec.pt/)) special session track. <br>
This work was done by [Ali Ismail-Fawaz](https://hadifawaz1999.github.io/), [Maxime Devanne](https://maxime-devanne.com/), [Stefano Berretti](www.micc.unifi.it/berretti/), [Jonathan Weber](https://www.jonathan-weber.eu/) and [Germain Forestier](https://germain-forestier.info/).

## The LITE architecture

The same LITE architecture is then used to form an ensemble of five LITE models, names LITETime.
<p align="center" width="50%">
<img src="images/LITE.png" alt="lite"/>
</p>

## Usage of the code

To use this code, running the ```main.py``` file with the following options in the parsed as arguments:

```
--dataset : to choose the dataset from the UCR Archive (default=Coffee)
--classifier : to choose the classifier, in this case only LITE can be chosen (default=LITE)
--runs : to choose the number of runs (default=5)
--output-directory : to choose the output directory name (default=results/)
--track-emissions : to choose wether or not to track the training/testing time and CO2/power consumotion.
```

### Adaptation of code

The only change to be done in the code is the ```folder_path``` in the `utils/utils.py` file, this line. This directory should point to the parent directory of the UCR Archive datasets.

## Results

### Average performance and FLOPS comparison

The following figure shows the comparison between LITE and state of the art complex deep learners. The comparison consists on the average performance and the number FLOPS.
<p align="center" width="100%">
<img src="images/summary_with_flops.png" alt="flops"/>
</p>

### LITE 1v1 with FCN, ResNet and Inception

### LITETime 1v1 with ROCKET and InceptionTime

### LITETime MCM with SOTA

### CD Diagram

## Requirements

```
numpy
pandas
sklearn
tensorflow
matplotlib
codecarbon
```

## Citation

```
@inproceedings{Ismail-Fawaz2023LITELightInception,
  author = {Ismail-Fawaz, A. and Devanne, M. and Berretti, S. and Weber, J. and Forestier, G.},
  title = {LITE: Light Inception with boosTing tEchniques for Time Series Classification},
  booktitle = {International Conference on Data Science and Advanced Analytics (DSAA)},
  year = {2023}
}
```






