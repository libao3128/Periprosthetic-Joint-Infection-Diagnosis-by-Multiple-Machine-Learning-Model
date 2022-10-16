# Periprosthetic Joint Infection Diagnosis by Multiple Machine Learning Model.
- Author: Li-Chun Huang
- Institute: Department of Computer Science at National Yang Ming Chiao Tung University
- Email: libao3128.cs08@nycu.edu.tw

## Introduction
Total knee/hip joint replacement (total knee/hip arthroplasty) is performed to restore function and relieve pain in patients with severely damaged knees. The surgery involves replacement of both the medial and lateral femorotibial joints and the patellofemoral joint. Although total joint replacement is an effective treatment, postoperative complications include blood clots, infection, and loosening or malalignment of the prosthetic component. Periprosthetic joint infection (PJI) is a serious complication occurring in 1% to 2% of primary arthroplasties, which is associated with high morbidity and need for complex interdisciplinary treatment strategies.

## Objective
This project aim to predict the PJI infection by given dataset. I have tried many differnet models to compare the performance of each. Besides, Also, since the data is used for medical application, it is important for us to make the model comprehensible.

## Data

The dataset is provided by Professor Yuh-Jyh Hu at Institute of Bioinformatics at National Yang Ming Chiao Tung University. The data is the real patients information collected from hosipital. This dataset is originally used for doctors to diagnosis whether the patient will get PJI or not. The dataset that used for PJI diagnosis contain 52159 samples, 67 features and 1 label. For more inofotmation about dataset, please see **Report.html**.
> The dataset is not public. If you want to access the whole dataset, please contact me at liabo3128.cs08@nycu.edu.tw or Professor Yuh-Jyu Hu at  yhu@cs.nycu.edu.tw

## Data Preprocessing
In order to get better performance, I have apply the following data preprocessing method.
- Remove low quality samples
- Undersampling
- Feature Selection
- Missing Value Imputation
- Scaler
- One hot Encoding
For more inofotmation about dataset, please see **Report.html**.

## Model
In this project, I have tried the following machine learning models and copare the result.
- CNN
- ANN
- Logistic Regression(LR)
- Support Vector Machine(SVM)
- Decision Tree(DT)
- Random Forest(RF)

Since the target dataset is extremely imbalanced, most of the model’s class weight are tuned as {0:0.1,1:1}.
For more inofotmation about dataset, please see **Report.html**.

## Results
Since the label in the original dataset is extremely imbalanced, accuracy in this case would not be a good way to estimate the model. Therefore, I used the addtional criteria below to compare the performance of each model.


| Performance Metric       | Definition                                                |
| ------------------------ | --------------------------------------------------------- |
| Percentage Accuracy(ACC) | (TP + TN) / (TP + TN + FP + FN)                           |
| Recall                   | TP / (TP + FN)                                            |
| Precision                | TP / (TP + FP)                                            |
| F1-score                 | $\frac{2\times Recall\times Precision}{Recall+Precision}$ |
| Matthews Correlation Coefficient (MCC)                  |  $\frac{TP\times TN - FP \times FN}{\sqrt{(TP+FP)\times(TP+FN)\times(TN+FP)\times(TN+FN)}}$ |

The performance below is test on the test dataset which is exclusive from train dataset.


| Model | Confusion matrix           | ACC    | Recall | Precision | F1-score | MCC    |
| ----- | -------------------------- | ------ | ------ | --------- | -------- | ------ |
| CNN   | [5368, 62]<br>[54, 107]    | 0.9792 | 0.6645 | 0.6331    | 0.6484   | 0.6380 |
| ANN   | [5113, 317]<br>[47, 114]   | 0.9348 | 0.7080 | 0.2645    | 0.3851   | 0.4073 |
| LR    | [3669, 1761]<br> [61, 100] | 0.6741 | 0.6211 | 0.0537    | 0.0989   | 0.1053 |
| SVM   |  [4969, 461]<br> [48, 113]          |  0.9089   |0.7018 |  0.1968         | 0.3074     | 0.3399 |
| DT      | [5350, 80]<br>  [61, 100]   | 0.9747  | 0.6211  | 0.5555  |     0.5865     | 0.5744   |
| RF      | [5419, 11]<br>  [59, 102]  | 0.9874| 0.6335 | 0.9026  | 0.7445   | 0.7504 |
