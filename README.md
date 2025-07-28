# CLAMSurvival
Adaptation of CLAM for multi-modal survival prediction with stain color normalization/augmentation.

## Purpose
The purpose of this repository is a faithful adaptation of CLAM for multi-modal deep learning-based survival analysis from clinical data and whole-slide imaging. A survival analysis adaptation of CLAM has already been published as BEPH, however this repository extends that code by three main aspects:
- integration of state-of-the-art stain normalization/augmentation techniques in feature extraction
- multi-modality: inclusion of clinical and image features via early fusion
- site and feature preserving data splitting by Howard et al.

This code is part of a research project titled "Impact of Stain Normalization on Computer-Aided Survival Prognosis using HE-Stained Whole-Slide Images in Breast Cancer" which was conducted under the supervision of Prof. Windberger from the Heilbronn University of Applied Sciences, Germany in cooperation with the Computation Biomedicine research group at the University of Turin, Italy with Prof. Fariselli and Prof. Sanavia. The resulting research [manuscript](Research_Project__Impact_of_Stain_Normalization.pdf) is published in this repository as well.

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
Whole-image slides were obtained from the The Cancer Genome Atlas Program (TCGA) Breast Invasive Carcinoma (BRCA) project through the [Genomic Data Commons (GDC) Data Portal](https://portal.gdc.cancer.gov/). Corresponding clinical data was retrieved from [cBioPortal](https://www.cbioportal.org/). The clinical data of the TCGA-BRCA cohort is listed in this [file](cbioportal_firehose_brca_tcga_clinical_data.tsv). The pre-processing of this data is conducted in [clinical_preprocessing.ipynb](clinical_preprocessing.ipynb). The resulting pre-processed clinical CSVs are located in dataset_csv/. More on the distinction between [pre-clinical](/CLAM/dataset_csv/tcga-brca-survival-pre-clinical.csv) and [diagnostic clinical](/CLAM/dataset_csv/tcga-brca-survival-clinical.csv) data can be read in the manuscript. 

## Pipeline
The survival analysis can be performed using clinical data alone or in unison with features extracted from whole slide imaging.
In order to perform survival analysis, these main steps must be followed:

1. Clinical data preprocessing
2. Data split creation
3. Patch creation*
4. Feature extraction*
5. Training*

* These steps are part of CLAM and best understood by consulting their README. Only the basic extended usage of this project will be explained here.

### Clinical data preprocessing
See [clinical_preprocessing.ipynb](clinical_preprocessing.ipynb).

### Data split creation
See [generate_cv.ipynb](/PreservedSiteCV/generate_cv.ipynb).

### Patch creation
The patching of the WSIs has to only be conducted once.
```
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset tcga.csv --seg --patch --stitch
```
DATA_DIRECTORY: path to directory containing WSIs
RESULTS_DIRECTORY: path to desired output directory

### Feature extraction
The feature extraction works just as in the original CLAM, with the exception of the choice of a --normalization_technique. Possible techniques are:
- Reinhard
- Vahadane
- Macenko
- Tellez (augmentation)
- [StainGAN](https://github.com/xtarx/StainGAN)
- [MultiStain CycleGAN](https://github.com/DBO-DKFZ/multistain_cyclegan_normalization)
- [HistAuGAN](https://github.com/sophiajw/HistAuGAN)

A possible feature extraction command could look like (replace TODOs):
```
CUDA_VISIBLE_DEVICES=TODO OMP_NUM_THREADS=TODO python extract_features_fp.py --data_h5_dir TODO --data_slide_dir TODO --csv_path TODO --feat_dir TODO --batch_size 512 --slide_ext .svs --normalization_technique TODO
```
data_h5_dir: RESULTS_DIRECTORY of patch creation
data_slide_dir: path to directory containing WSIs
csv_path: contains a list of slide filenames
feat_dir: path to desired output directory
normalization_technique: "reinhard","macenko","vahadane","staingan","tellez_augmentation","multistain_cyclegan","histaugan"

### Training and Testing
Training and testing are both done using main.py.
If the multi-modal model should be run, i.e. using clinical data as well as WSI features, this command should be used:
```
CUDA_VISIBLE_DEVICES=TODO python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code clinical_and_image_earlyfusion --weighted_sample --bag_loss ce --inst_loss mse --task survival_prediction --model_type clam_sb --log_data --data_root_dir TODO --embed_dim 1024 --split_dir TODO --include_clinical_data --clinical_input_dim TODO
```

If only clinical data is of interest, this command should be used:
```
CUDA_VISIBLE_DEVICES=TODO python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code clinical_and_image_earlyfusion --weighted_sample --bag_loss ce --inst_loss mse --task survival_prediction --model_type clam_sb --log_data --data_root_dir TODO --embed_dim 1024 --split_dir TODO --clinical_only
```
csv_file_name: name of csv file located under dataset_csv/ containing slide_id, label and duration columns. All clinical columns must have the prefix 'clinical_' and should be scaled in range 0 to 1.
data_root_dir: directory containing the extracted features (feat_dir of extract_features_fp.py)
split_dir: directory containing split csv files (see /CLAM/splits/)
clinical_input_dim: number of columns with 'clinical_' prefix.
