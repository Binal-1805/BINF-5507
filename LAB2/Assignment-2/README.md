# Assignment 2 - Machine Learning

This project involves building both regression and classification models using the UCI Heart Disease dataset. The goal is to predict cholesterol levels (regression) and heart disease presence (classification).


## Files
- **Data/heart_disease_uci.csv**: The provided dataset.
- **Script/main.ipynb**: Jupyter notebook for implementing the models.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize
