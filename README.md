# CLAMSurvival
Adaptation of CLAM for multi-modal survival prediction with stain color normalization/augmentation.

## Purpose
The purpose of this repository is a faithful adaptation of CLAM for multi-modal deep learning-based survival analysis from clinical data and whole-slide imaging. A survival analysis adaptation of CLAM has already been published as BEPH, however this repository extends that code by three main aspects:
- integration of state-of-the-art stain normalization/augmentation techniques in feature extraction
- multi-modality: inclusion of clinical and image features via early fusion
- site and feature preserving data splitting by Howard et al.

This code is part of a research project titled "Impact of Stain Normalization on Computer-Aided Survival Prognosis using HE-Stained Whole-Slide Images in Breast Cancer" which was conducted under the supervision of Prof. Windberger from the Heilbronn University of Applied Sciences, Germany in cooperation with the Computation Biomedicine research group at the University of Turin, Italy with Prof. Fariselli and Prof. Sanavia. The resulting research [manuscript]() is published in this repository as well.

## Acknowledgements

[CLAM on GitHub](https://github.com/mahmoodlab/CLAM)

```
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group}
}
```

[BEPH on GitHub](https://github.com/Zhcyoung/BEPH)

```
@article{yang_foundation_2025,
	title = {A foundation model for generalizable cancer diagnosis and survival prediction from histopathological images},
	author = {Yang, Zhaochang and Wei, Ting and Liang, Ying and Yuan, Xin and Gao, RuiTian and Xia, Yujia and Zhou, Jie and Zhang, Yue and Yu, Zhangsheng},
	url = {https://doi.org/10.1038/s41467-025-57587-y},
	journal = {Nature Communications},
	year = {2025},
}
```
## Data acquisition and preprocessing
Whole-image slides were obtained from the The Cancer Genome Atlas Program (TCGA) Breast Invasive Carcinoma (BRCA) project through the [Genomic Data Commons (GDC) Data Portal](https://portal.gdc.cancer.gov/). Corresponding clinical data was retrieved from [cBioPortal](https://www.cbioportal.org/). The clinical data of the TCGA-BRCA cohort is listed in this [file](cbioportal_firehose_brca_tcga_clinical_data.tsv). The pre-processing of this data is conducted in [clinical_preprocessing.ipynb](clinical_preprocessing.ipynb). The resulting pre-processed clinical CSVs are located in dataset_csv/. More on the distinction between [CLAM/dataset_csv/tcga-brca-survival-pre-clinical.csv](pre-clinical) and [CLAM/dataset_csv/tcga-brca-survival-clinical.csv](diagnostic clinical) data can be read in the manuscript. 

## Pipeline
The survival analysis can be performed using clinical data alone or in unison with features extracted from whole slide imaging.
In order to perform survival analysis, these main steps must be followed:

1. Clinical data preprocessing
2. Patch creation
3. Data split creation
4. Feature extraction
4. Training
