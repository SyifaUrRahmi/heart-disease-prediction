# Heart Disease Prediction : A case study of 2020 annual CDC survey data of 400k adults related to their health status

## Abstract

Heart disease is a condition when the heart is disturbed. 1 of the 3 main risk factors for heart disease: high blood pressure, high cholesterol, and smoking. Other key indicators include diabetes status, obesity (high BMI), lack of physical activity or drinking too much alcohol. Detecting and preventing the factors that have the greatest impact on heart disease is very important in health care. The purpose of this writing is to predict whether someone has heart disease or not using the 2020 annual CDC survey data of 400k adults related to their health status. The data is then predicted using a machine learning approach, namely decision trees, svm, and knn involving 17 variables as risk factors.

## Processing Data

IMPORTING LIBRARIES AND LOADING DATA
```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import missingno as msno
from scipy import stats
```
```python
data = pd.read_csv('../csv/heart_2020_cleaned.csv')
```


