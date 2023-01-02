# Human Protein Atlas Image Classification
## Installation
```
git clone https://github.com/florinhegedus/protein_classification.git
conda create --name protein_classification --file requirements.txt
conda activate protein_classification
```

## Run
Download the Human Protein Atlas Image Classification Dataset from [kaggle](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/data).
Update the dataset path in config.yaml to match the one on your system.

### Oversample the dataset (optionally)
Run the oversample script to duplicate rare classes in the dataset:
```
python oversample.py config.yaml
```

### Train
Change the annotations file path in config.yaml if it is the case.
```
python train.py config.yaml
```

### Evaluate
Generate the sample_submission.csv file (predictions of the model on the test set:
```
python evaluate.py config.yaml
```