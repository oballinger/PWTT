# Pixel-Wise T-Test

## A new algorithm for battle damage detection using Sentinel-1 imagery 

This [paper](https://arxiv.org/pdf/2405.06323) develops a new algorithm for the detection of urban conflict damage– the Pixelwise T-Test (PWTT)– that is lightweight, unsupervised, and uses only freely-available Synthetic Aperture Radar (SAR) imagery. 

## Validation Data 

By joining manual damage annotations from UNOSAT with building footprints, 700,488 labeled building footprints spread across 12 cities in 4 different conflict zones are used for validation.The dataset can be downloaded [here](https://drive.google.com/file/d/1AjsCJ5Wc0xDRUcc0VHtATLWee2lVN8RM/view?usp=sharing).

<h3 align="center">Validation Dataset</h3>

| Country   | City        | Footprints | Percent Damaged | Annotation Date |
|-----------|-------------|-------------|-----------------|-----------------|
| Palestine | Gaza        | 220160      | 57.21%          | 2024-05-03      |
| Ukraine   | Lysychansk  | 22280       | 7.77%           | 2022-09-21      |
| Ukraine   | S'odonetsk  | 7085        | 25.43%          | 2022-07-27      |
| Ukraine   | Rubizhne    | 9905        | 33.81%          | 2022-07-09      |
| Ukraine   | Kharkiv     | 122590      | 0.91%           | 2022-06-15      |
| Ukraine   | Mariupol    | 20481       | 32.57%          | 2022-05-12      |
| Ukraine   | Hostomel    | 4728        | 14.11%          | 2022-03-31      |
| Ukraine   | Irpin       | 7917        | 11.56%          | 2022-03-31      |
| Ukraine   | Chernihiv   | 33733       | 3.36%           | 2022-03-22      |
| Syria     | Raqqa       | 26728       | 45.78%          | 2017-10-21      |
| Syria     | Aleppo      | 73634       | 31.45%          | 2016-09-18      |
| Iraq      | Mosul       | 151259      | 11.59%          | 2017-08-04      |


## Quickstart

git clone https://github.com/oballinger/PWTT
cd code

