<div align="center">

<h1 align="center">Pixel-Wise T-Test</h1>
<h2 align="center">A New Algorithm for Battle Damage Detection using Sentinel-1 Imagery </h2>

[![](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/pdf/2405.06323)  [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oballinger/PWTT/blob/main/pwtt_quickstart.ipynb)   [![](https://img.shields.io/badge/Benchmark%20Dataset-8A2BE2)](https://drive.google.com/file/d/12RsrfU8m-cvtONohD6FBcF21OPoEA_Mf/view?usp=sharing)

</div>

![](figs/pwtt_pre_post.png)

The generation of public information regarding buildings damaged by conflict has become particularly important in the context of recent, highly destructive wars in Gaza and Ukraine. This paper develops the Pixelwise T-Test (PWTT), a new algorithm for open-access battle damage estimation that is accurate, lightweight, and generalizable. The PWTT addresses many of the problems associated with expense, coverage consistency, and domain shift that affect deep-learning based approaches to building damage detection, and outperforms the state of the art deep learning model on unseen areas. 

## How it Works 

The PWTT utilizes Synthetic Aperture Radar imagery from the Sentinel-1 satellite. The figure below demonstrates the change in backscatter amplitude (i.e., the "loudness" of the signal's echo) for a destroyed building in Mariupol, Ukraine, before and after its destruction. The corresponding Sentinel-1 pixel has a low standard deviation in both the pre-and post-war periods, but experiences a large change in mean amplitude. The T-Test is a simple signal-to-noise ratio that measures the difference between the means of two samples adjusted by the standard deviation within each sample.

![](figs/single_building.png)

The green dashed line and shaded area represent the pixel's mean backscatter amplitude $\pm$ 1 standard deviation prior to the invasion, while the red line and shaded area represent these statistics following the building's destruction.

## Python Quickstart [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oballinger/PWTT/blob/main/pwtt_quickstart.ipynb)


First, clone the repository and navigate to the code directory 

```python
!git clone https://github.com/oballinger/PWTT
!cd PWTT/code
```

Then, import the Google Earth Engine python API and authenticate using your credentials and cloud project name. 

```python
import ee
import pwtt

project_name='<YOUR PROJECT NAME>'
ee.Authenticate()
ee.Initialize(project=project_name)
```

finally, the PWTT can be deployed in one line of code. The example below conducts a damage assessment over Gaza for July 2024.

```python
gaza = ee.Geometry.Rectangle([34.21,31.21,34.57,31.60])

pwtt.filter_s1(aoi=gaza, # the area of interest as a bounding box
                   war_start='2023-10-10', # the start of the war
                   inference_start='2024-07-01', # the beginning of the inference window
                   pre_interval=12, # the number of months before the war to use as a reference period 
                   post_interval=1, # the number of months after the war to use as an inference period 
                   viz=True) # visualize the results
```

By simply modifying the location, war_start date and inference_start dates, damage assessment can be carried out on a new area; below is another example for Bakhmut, Ukraine:

```python
bakhmut = ee.Geometry.Rectangle([37.949421, 48.556181, 38.043834, 48.621584])

pwtt.filter_s1(aoi=bakhmut,
                   war_start='2022-02-22',
                   inference_start='2024-07-01',
                   pre_interval=12,
                   post_interval=1,
                   viz=True)
```

## Validation Data 

Accuracy assessments are carried out using an original dataset of 700,500 annotated building footprints, spanning 12 cities in four different countries. This dataset was compiled by spatially joining damage annotations from the United Nations Satellite Centre (UNOSAT) with data on building footprints. The dataset can be downloaded [here](https://drive.google.com/file/d/1AjsCJ5Wc0xDRUcc0VHtATLWee2lVN8RM/view?usp=sharing).

<h3 align="center">Benchmark Dataset</h3>

| Country   | City              | Footprints | Percent Damaged | Annotation Date |
|-----------|--------------------|------------|------------------|------------------|
| Palestine | Gaza              | 228728     | 57.49%           | 2024-05-03       |
| Ukraine   | Rubizhne          | 8899       | 33.67%           | 2022-07-09       |
| Ukraine   | Mariupol          | 18446      | 31.42%           | 2022-05-12       |
| Ukraine   | Sievierodonetsk   | 5970       | 24.32%           | 2022-07-27       |
| Ukraine   | Hostomel          | 4175       | 14.04%           | 2022-03-31       |
| Ukraine   | Irpin             | 7242       | 11.28%           | 2022-03-31       |
| Ukraine   | Avdiivka          | 7262       | 8.7%             | 2022-09-20       |
| Ukraine   | Lysychansk        | 20246      | 7.42%            | 2022-09-21       |
| Ukraine   | Makariv           | 3514       | 4.01%            | 2022-03-16       |
| Ukraine   | Chernihiv         | 29929      | 3.25%            | 2022-03-22       |
| Ukraine   | Shchastia         | 1293       | 1.93%            | 2022-07-07       |
| Ukraine   | Bucha             | 5739       | 1.6%             | 2022-03-31       |
| Ukraine   | Kharkiv           | 107976     | 0.85%            | 2022-06-15       |
| Ukraine   | Trostianets       | 8913       | 0.72%            | 2022-03-25       |
| Ukraine   | Okhtyrka          | 15905      | 0.4%             | 2022-03-25       |
| Ukraine   | Mykolaiv          | 60467      | 0.34%            | 2022-07-21       |
| Ukraine   | Melitopol         | 32373      | 0.23%            | 2022-08-02       |
| Ukraine   | Kramatorsk        | 21880      | 0.22%            | 2022-07-24       |
| Ukraine   | Kremenchuk        | 27497      | 0.11%            | 2022-06-29       |
| Ukraine   | Sumy              | 28265      | 0.08%            | 2022-07-08       |
| Syria     | Raqqa             | 24689      | 44.67%           | 2017-10-21       |
| Syria     | Aleppo            | 65870      | 26.65%           | 2016-09-18       |
| Iraq      | Mosul             | 137794     | 11.51%           | 2017-08-04       |
| **All**   | **All**           | 873072     | **12.39%**       |                  |


Building footprint data is sourced from the Microsoft Building Footprints dataset, which consists of over 1 billion building footprints derived from high resolution satellite imagery around the world. A building footprint is labeled as damaged if it intersects with a UNOSAT damage annotation point, and labeled undamaged otherwise. UNOSAT annotations are generated manually on the basis of high resolution optical satellite imagery.

## Accuracy Assessment

The table below reports the accuracy statistics for the PWTT algorithm in 12 cities, assessed using the benchmark dataset above. 

| Country   | City               | AUC   | Accuracy | F1    | Precision | Recall | N       |
|-----------|---------------------|-------|----------|-------|-----------|--------|---------|
| Palestine | Gaza               | 83.32 | 75.41    | 78.2  | 70.42     | 87.91  | 228728  |
|-----------|---------------------|-------|----------|-------|-----------|--------|---------|
| Ukraine   | Kremenchuk         | 92.56 | 98.82    | 2.73  | 2.68      | 2.77   | 27497   |
| Ukraine   | Bucha              | 90.19 | 98.19    | 71.9  | 98.27     | 56.68  | 5739    |
| Ukraine   | Okhtyrka           | 89.28 | 98.9     | 53.14 | 76.28     | 40.77  | 15905   |
| Ukraine   | Kramatorsk         | 88.24 | 97.16    | 7.92  | 11.99     | 5.91   | 21880   |
| Ukraine   | Trostianets        | 86.63 | 98.44    | 67.6  | 80.58     | 58.23  | 8913    |
| Ukraine   | Chernihiv          | 86.61 | 94.47    | 48.29 | 53.48     | 44.03  | 29929   |
| Ukraine   | Hostomel           | 84.99 | 80.44    | 68.11 | 63.87     | 72.95  | 4175    |
| Ukraine   | Irpin              | 83.81 | 82.93    | 56.75 | 52.83     | 61.3   | 7242    |
| Ukraine   | Kharkiv            | 83.78 | 97.24    | 29.37 | 30.44     | 28.36  | 107976  |
| Ukraine   | Mykolaiv           | 83.28 | 98.18    | 21.61 | 18.71     | 25.57  | 60467   |
| Ukraine   | Makariv            | 82.87 | 90.37    | 40.69 | 34.88     | 48.82  | 3514    |
| Ukraine   | Lysychansk         | 77.91 | 82.59    | 50.14 | 47.38     | 53.24  | 20246   |
| Ukraine   | Rubizhne           | 77.81 | 71.38    | 66.42 | 62.14     | 71.33  | 8899    |
| Ukraine   | Mariupol           | 73.85 | 66.17    | 71.08 | 60.78     | 85.57  | 18446   |
| Ukraine   | Shchastia          | 73.39 | 96.69    | 38.31 | 47.12     | 32.28  | 1293    |
| Ukraine   | Sumy               | 69.63 | 99.51    | 4.26  | 4.36      | 4.16   | 28265   |
| Ukraine   | Sievierodonetsk    | 69.18 | 59.3     | 62.53 | 50.71     | 81.54  | 5970    |
| Ukraine   | Melitopol          | 66.84 | 96.77    | 2.67  | 1.78      | 5.33   | 32373   |
| Ukraine   | Avdiivka           | 66.51 | 60.64    | 39.8  | 28.05     | 68.5   | 7262    |
|-----------|---------------------|-------|----------|-------|-----------|--------|---------|
| Syria     | Raqqa              | 75.6  | 66.26    | 73.87 | 63.21     | 88.87  | 24689   |
| Syria     | Aleppo             | 71.76 | 61.49    | 58.82 | 46.28     | 80.67  | 65870   |
|-----------|---------------------|-------|----------|-------|-----------|--------|---------|
| Iraq      | Mosul              | 74.91 | 81.16    | 41.18 | 35.18     | 49.64  | 137794  |
|-----------|---------------------|-------|----------|-------|-----------|--------|---------|
| **All**   | **All**            | 84.17 | 76.4     | 58.2  | 48.04     | 73.82  | 873072  |


Reciever-Operating Characteristic (ROC) curves for each country are also provided below.
![](figs/roc.png)