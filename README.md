# CytoCoSet

#[![CircleCI](https://circleci.com/gh/CompCy-lab/cytoset.svg?style=svg&circle-token=7070f7f23c7fccba6d452bcfd2ee2a1cb469a6e0)](https://circleci.com/gh/CompCy-lab/cytoset)

## Introduction

CytoCoSet is a set-based encoding method, which formulates a loss function with an additional triplet term penalizing samples with similar covariates from having disparate embeddings results in per-sample representations.

<p align="center">
<img align="middle" src="./assets/overview.png" alt="CytoCoSet" width="600" />
</p>

## Installation

### Requirements

- Python >= 3.6
- CUDA >= 10.1

```
pip install -r requirements.txt
```

### Datasets

-   [preeclampsia](https://zenodo.org/record/6779483#.Yrygu-zMJhF)
-   [covid](https://zenodo.org/record/6780354#.Yryxg-zMJhE)
-   [NK cell](https://zenodo.org/record/6780417#.Yry12-zMJhE)


## Reproducing Results

### Training

* Download pre-processed the datasets (see Datasets Section) and unpack them.
* In ``scripts/train/train_[Dataset].sh``, set ``bin_file`` to the path of ``train.py`` and ``gpu`` to the gpu id.
* Start training: ``bash train_[Dataset].sh``


### Testing
* We provide our pre-trained model on HVTN dataset and test dataset in ``checkpoints``.
* We also provide our model configuration for each dataset in ``config/model``.
* To run the testing, you can use the following command:
```
python test.py --model checkpoints/HVTN_model.pt --config config/model/ICS/config.json --test_pkl checkpoints/test_sample.pkl
```
The evaluation results are:
| Accuracy  | Area Under Curve |
|-----------|------------------|
|   0.958   |     0.962        |


## Citing

```
@inproceedings{
    10.1145/3459930.3469529,
    author = {Yi, Haidong and Stanley, Natalie},
    title = {CytoSet: Predicting Clinical Outcomes via Set-Modeling of Cytometry Data},
    year = {2021},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3459930.3469529}
}
```


## Contact

If you have any questions, please feel free to contact Haidong Yi (haidyi@cs.unc.edu) or push an issue on Issues Dashboard.


