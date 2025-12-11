# RESULTS

## 1. Dataset Loading

Loaded clean dataset: 7994 rows × 22 columns

### 1.1 DataFrame Information

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7994 entries, 0 to 7993
Data columns (total 22 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   kepid             7994 non-null   int64  
 1   kepoi_name        7994 non-null   object 
 2   kepler_name       2729 non-null   object 
 3   koi_score         7994 non-null   float64
 4   koi_disposition   7994 non-null   object 
 5   koi_pdisposition  7994 non-null   object 
 6   koi_fpflag_nt     7994 non-null   int64  
 7   koi_fpflag_ss     7994 non-null   int64  
 8   koi_fpflag_co     7994 non-null   int64  
 9   koi_fpflag_ec     7994 non-null   int64  
 10  koi_period        7994 non-null   float64
 11  koi_impact        7994 non-null   float64
 12  koi_duration      7994 non-null   float64
 13  koi_depth         7994 non-null   float64
 14  koi_model_snr     7994 non-null   float64
 15  koi_prad          7994 non-null   float64
 16  koi_teq           7994 non-null   float64
 17  koi_insol         7994 non-null   float64
 18  koi_steff         7994 non-null   float64
 19  koi_slogg         7994 non-null   float64
 20  koi_srad          7994 non-null   float64
 21  koi_kepmag        7994 non-null   float64
dtypes: float64(13), int64(5), object(4)
memory usage: 1.3+ MB
```

### 1.2 Summary Statistics

```
           koi_score   koi_period  koi_impact  koi_duration      koi_depth  koi_model_snr      koi_prad       koi_teq     koi_insol     koi_steff   koi_slogg     koi_srad   koi_kepmag
count  7994.000000  7994.000000  7994.00000   7994.000000    7994.000000    7994.000000   7994.000000   7994.000000  7.994000e+03   7994.000000  7994.000000  7994.000000  7994.000000
mean      0.483829    37.117158     0.61885      5.370080   26648.655679     294.795334     27.918302   1142.014761  8.124302e+03   5691.418064     4.319325     1.713651    14.317879
std       0.477009    86.677491     0.77548      6.309563   85048.286984     845.134235    312.092199    846.297825  1.689601e+05    788.322057     0.424413     6.422343     1.366927
min       0.000000     0.259820     0.00000      0.052000       0.000000       0.000000      0.080000     92.000000  2.000000e-02   2661.000000     0.047000     0.109000     6.966000
25%       0.000000     2.421038     0.21600      2.414150     162.400000      14.300000      1.410000    612.000000  3.316000e+01   5312.000000     4.229000     0.827000    13.506250
50%       0.371000     7.582527     0.58250      3.733650     449.150000      27.500000      2.470000    938.000000  1.833250e+02   5761.000000     4.438000     0.997000    14.575000
75%       0.999000    23.815564     0.90900      5.959750    1996.200000     100.075000     19.702500   1435.000000  1.003577e+03   6098.000000     4.544000     1.316000    15.341750
max       1.000000  1071.232624    25.22400    138.540000  921670.000000    9054.700000  26042.900000  14667.000000  1.094755e+07  15896.000000     5.364000   229.908000    20.003000
```

Number of exact duplicate rows: 0

## 2. Quality Labeling

After applying koi_score thresholds:

- Labeled samples: 7735
- Label distribution (0=low quality, 1=high quality):
    - quality_label 0: 3948
    - quality_label 1: 3787

### 2.1 Feature Transformations

Using fixed transforms per FRAMEWORK:

- **Log10 columns:** ['koi_period', 'koi_duration', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_srad']
- **Log10(1+x) columns:** ['koi_depth', 'koi_model_snr']
- **Linear columns:** ['koi_impact', 'koi_steff', 'koi_slogg', 'koi_kepmag']

## 3. Train-Test Split

- Training set (standardized): (6188, 12)
- Test set (standardized): (1547, 12)

**Training label distribution:**

- quality_label 0: 3158
- quality_label 1: 3030

**Test label distribution:**

- quality_label 0: 790
- quality_label 1: 757

## 4. Principal Component Analysis

### 4.1 Eigenvalues and Explained Variance

PCA on training set:

```
Component  Eigenvalue  Explained_%  Cumulative_%
PC1        3.735430       31.13        31.13
PC2        2.995335       24.96        56.09
PC3        2.082775       17.36        73.45
PC4        1.002589        8.35        81.80
PC5        0.922669        7.69        89.49
PC6        0.652634        5.44        94.93
PC7        0.432312        3.60        98.53
PC8        0.116624        0.97        99.50
PC9        0.052993        0.44        99.94
PC10       0.005866        0.05        99.99
PC11       0.000771        0.01       100.00
```

### 4.2 Loadings Matrix A (First 5 PCs)

```
Component  koi_period  koi_impact  koi_duration  koi_depth  koi_model_snr  koi_prad  koi_teq  koi_insol  koi_steff  koi_slogg  koi_srad  koi_kepmag
PC1          -0.3130      0.0763       -0.0908     0.0405         0.1397   -0.1825  -0.3823    -0.3809    -0.2831     0.3529   -0.3737     -0.2562
PC2           0.2975      0.1918        0.3598     0.4760         0.4324    0.4754   0.0711     0.0936    -0.0449    -0.0267    0.0876      0.2799
PC3           0.3937     -0.1590        0.3120    -0.3283        -0.3064   -0.3152   0.2197     0.2186     0.1754    -0.3302    0.3362      0.2997
PC4           0.0260      0.2088       -0.2170     0.0120        -0.1425    0.0536  -0.2042    -0.1995     0.8532     0.0044   -0.2175      0.1434
PC5           0.0295      0.9023        0.0477    -0.2319        -0.2556   -0.1829  -0.0020    -0.0045     0.0451     0.0166    0.0043      0.0925
PC6          -0.0030      0.1464       -0.3315     0.0194         0.2218    0.0733   0.0637     0.0639    -0.0329    -0.6799    0.0635      0.6045
PC7           0.3141     -0.0570       -0.7704     0.1473        -0.1120    0.0868   0.2031     0.2014    -0.1157     0.0396    0.2079     -0.3088
PC8           0.1082      0.1984       -0.0712    -0.0612         0.6140   -0.0331   0.0623     0.0600     0.1260    -0.4522    0.0607      0.5632
PC9          -0.0505      0.1018        0.0836     0.7638        -0.4184    0.3864   0.0251     0.0284    -0.0382     0.1653    0.0278     -0.1592
PC10         -0.0065     -0.0042       -0.0073    -0.0294         0.0104    0.0116  -0.7050     0.7068    -0.0115     0.0037   -0.0007      0.0067
PC11         -0.7376      0.0009        0.0016    -0.0077         0.0063    0.0027  -0.3745     0.3739    -0.3344    -0.0005    0.3327      0.0009
```

Retaining k=4 principal components, covering 81.80% of variance.

PCA feature matrices → X_train_pca: (6188, 4), X_test_pca: (1547, 4)

### 4.3 Top Contributors by Component

- **Top contributors to PC1 (by |loading|):** koi_teq, koi_insol, koi_srad, koi_slogg, koi_period
- **Top contributors to PC2 (by |loading|):** koi_depth, koi_prad, koi_model_snr, koi_duration, koi_period
- **Top contributors to PC3 (by |loading|):** koi_period, koi_srad, koi_slogg, koi_kepmag, koi_depth

## 5. Prepared Feature Matrices

- x_train_original: (6188, 12), x_test_original: (1547, 12)
- x_train_pca: (6188, 4), x_test_pca: (1547, 4)
- PCA column names: ['PC1', 'PC2', 'PC3', 'PC4']

## 6. Model Comparison on ORIGINAL Features

Comparing models on ORIGINAL features:

```
                               Model  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC  TT (Sec)
Light Gradient Boosting Machine       0.8751  0.9417  0.8939  0.8576  0.8752  0.7503  0.7513     0.907
Extra Trees Classifier                0.8686  0.9364  0.8873  0.8516  0.8688  0.7374  0.7384     0.219
Gradient Boosting Classifier          0.8668  0.9338  0.8949  0.8436  0.8681  0.7338  0.7357     0.971
Random Forest Classifier              0.8684  0.9358  0.8798  0.8565  0.8677  0.7368  0.7376     0.681
K Neighbors Classifier                0.8601  0.9147  0.8902  0.8357  0.8619  0.7204  0.7223     0.034
Ada Boost Classifier                  0.8451  0.9133  0.8779  0.8195  0.8474  0.6905  0.6927     0.235
Decision Tree Classifier              0.8114  0.8113  0.8062  0.8084  0.8072  0.6225  0.6228     0.046
Ridge Classifier                      0.7747  0.8366  0.9269  0.7059  0.8012  0.5520  0.5802     0.010
Linear Discriminant Analysis          0.7747  0.8361  0.9269  0.7059  0.8012  0.5520  0.5802     0.011
Quadratic Discriminant Analysis       0.7737  0.8843  0.9222  0.7063  0.7997  0.5500  0.5768     0.012
Logistic Regression                   0.7818  0.8417  0.8755  0.7323  0.7972  0.5652  0.5764     0.028
Naive Bayes                           0.7585  0.8616  0.8916  0.6989  0.7833  0.5194  0.5399     0.010
SVM - Linear Kernel                   0.7613  0.8196  0.8076  0.7320  0.7656  0.5231  0.5296     0.018
Dummy Classifier                      0.5103  0.5000  0.0000  0.0000  0.0000  0.0000  0.0000     0.009
```

## 7. Model Comparison on PCA Features

Comparing models on PCA features:

```
                               Model  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC  TT (Sec)
Light Gradient Boosting Machine       0.8047  0.8774  0.8364  0.7810  0.8075  0.6097  0.6117     0.426
Gradient Boosting Classifier          0.8010  0.8694  0.8369  0.7756  0.8047  0.6024  0.6048     0.411
Random Forest Classifier              0.8026  0.8750  0.8157  0.7897  0.8019  0.6053  0.6065     0.522
Extra Trees Classifier                0.7994  0.8740  0.8157  0.7845  0.7994  0.5989  0.5999     0.215
K Neighbors Classifier                0.7945  0.8578  0.8076  0.7810  0.7938  0.5891  0.5899     0.018
Ada Boost Classifier                  0.7834  0.8517  0.8312  0.7531  0.7899  0.5676  0.5711     0.158
Quadratic Discriminant Analysis       0.7432  0.8483  0.9085  0.6775  0.7761  0.4898  0.5195     0.012
Naive Bayes                           0.7497  0.8289  0.8519  0.7017  0.7692  0.5014  0.5134     0.009
Ridge Classifier                      0.7409  0.7721  0.8680  0.6862  0.7664  0.4844  0.5018     0.009
Linear Discriminant Analysis          0.7409  0.7721  0.8680  0.6862  0.7664  0.4844  0.5018     0.009
Logistic Regression                   0.7336  0.7706  0.8260  0.6905  0.7520  0.4690  0.4783     0.015
Decision Tree Classifier              0.7361  0.7358  0.7232  0.7349  0.7284  0.4718  0.4727     0.023
SVM - Linear Kernel                   0.7095  0.7455  0.7901  0.6749  0.7244  0.4207  0.4336     0.016
Dummy Classifier                      0.5103  0.5000  0.0000  0.0000  0.0000  0.0000  0.0000     0.008
```

## 8. Model Training and Tuning

### 8.1 Training and Tuning Models on ORIGINAL Features

**LightGBM (Original)**

```
Metric  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold           -       -       -       -       -       -       -
Std       0.0145  0.0105  0.0169  0.0201  0.014   0.029  0.0288
```

**Gradient Boosting Classifier (Original)**

```
Metric  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold           -       -       -       -       -       -       -
Std       0.0161  0.0087  0.0138  0.0212  0.0153  0.0322  0.0319
```

**Random Forest (Original)**

```
Metric  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold           -       -       -       -       -       -       -
Std       0.0135  0.0141  0.0125  0.0237  0.0103  0.0266  0.0237
```

**Extra Trees Classifier (Original)**

```
Metric  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold           -       -       -       -       -       -       -
Std        0.017  0.0135  0.0196  0.0281  0.0146  0.0338  0.0327
```

**K-Neighbors Classifier (Original)**

```
Metric  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold           -       -       -       -       -       -       -
Std       0.0124  0.0149  0.0126  0.0196  0.0109  0.0246  0.0235
```

Tuned models on original features stored.

### 8.2 Training and Tuning Models on PCA Features

**LightGBM (PCA)**

```
Metric  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold           -       -       -       -       -       -       -
Std       0.0129  0.0143  0.0155  0.0214    0.01  0.0255  0.0242
```

**Gradient Boosting Classifier (PCA)**

```
Metric  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold           -       -       -       -       -       -       -
Std       0.0166  0.0147   0.015  0.0235   0.014   0.033  0.0316
```

**Random Forest (PCA)**

```
Metric  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold           -       -       -       -       -       -       -
Std       0.0163  0.0158  0.0199  0.0242  0.0135  0.0323   0.031
```

**Extra Trees Classifier (PCA)**

```
Metric  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold           -       -       -       -       -       -       -
Std       0.0182  0.0108  0.0161  0.0231  0.0153  0.0361  0.0348
```

**K-Neighbors Classifier (PCA)**

```
Metric  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold           -       -       -       -       -       -       -
Std        0.016   0.015   0.018  0.0207  0.0146  0.0318  0.0314
```

Tuned models on PCA features stored.

## 9. Test Set Evaluation

Evaluating models on test sets...

### 9.1 Model Comparison Summary (Test Set Performance)

```
Model      Features         Accuracy      F1  Precision  Recall     AUC
LightGBM   Original (12-D)    0.8836  0.8834     0.8666  0.9009  0.9413
LightGBM   PCA (4 PCs)        0.7957  0.8000     0.7679  0.8349  0.8621
GBC        Original (12-D)    0.8830  0.8830     0.8646  0.9022  0.9422
GBC        PCA (4 PCs)        0.7983  0.8033     0.7684  0.8415  0.8655
RF         Original (12-D)    0.8707  0.8701     0.8557  0.8851  0.9387
RF         PCA (4 PCs)        0.7951  0.8037     0.7564  0.8573  0.8634
ET         Original (12-D)    0.8694  0.8700     0.8482  0.8930  0.9361
ET         PCA (4 PCs)        0.7835  0.7926     0.7459  0.8454  0.8524
KNN        Original (12-D)    0.8746  0.8769     0.8437  0.9128  0.9237
KNN        PCA (4 PCs)        0.8093  0.8155     0.7743  0.8613  0.8745
```

**Best Model by F1 Score:** LightGBM on Original (12-D)

- F1 = 0.8834
- Accuracy = 0.8836
- AUC = 0.9413

## 10. Feature Importance Analysis

### 10.1 Top 5 Most Important Original Features

1. koi_prad: 0.2050
2. koi_depth: 0.1062
3. koi_period: 0.1061
4. koi_insol: 0.0952
5. koi_duration: 0.0917

### 10.2 Top 3 Most Important Principal Components

1. PC2: 0.3232
2. PC1: 0.2902
3. PC3: 0.2261

Feature importance analysis complete.