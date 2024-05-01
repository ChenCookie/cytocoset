# CytoCoSet

## Introduction

CytoCoSet is a set-based encoding method, which formulates a loss function with an additional triplet term penalizing samples with similar covariates from having disparate embeddings results in per-sample representations.

<p align="center">
<img align="middle" src="./assets/overview.png" alt="CytoCoSet" width="800" />
</p>

## Installation

### Requirements

- Python >= 3.6
- CUDA >= 10.1

```
pip install -r requirements.txt
```

### Datasets

Please download the dataset from Zenodo and follow the instruction of data structure that shown in Zenodo webpage description.



-   [Preeclampsia](https://doi.org/10.5281/zenodo.10659650)
-   [Preterm](https://doi.org/10.5281/zenodo.10660080)
-   [Lung Cancer](https://doi.org/10.5281/zenodo.10659930)


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



## Contact

If you have any questions or need further assistance, please don't hesitate to reach out to Chi-Jane Chen at chijane@cs.unc.edu, or simply submit an issue on the Issues Dashboard. Your inquiries are always welcome!


