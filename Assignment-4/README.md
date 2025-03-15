
=======
Survival Analysis of  Head and Neck Cancer Patients

This assignment analyzes clinical data from head and neck cancer patients to build and evaluated survival models. The analysis includes Kaplan–Meier survival curves, Cox Proportional Hazards regression, and Random Survival Forests (RSF) to predict survival outcomes and assess the predictive importance of various clinical factors.

Dataset
Data File: RADCURE_Clinical_v04_20241219.xlsx
Key Variables:
Survival time (Length FU)
Event indicator (Status, mapped to 0 for Alive and 1 for Dead)
Covariates such as Age, Stage, and Treatment Modality


Methodology
1. Kaplan–Meier Analysis:
Survival curves were generated for distinct patient groups (e.g., based on age groups).
Log-rank tests were used to statistically compare survival differences between groups.
Cox Proportional Hazards Regression:
A multivariable Cox model was fitted using Age, Stage, and Treatment Modality as predictors.
The proportional hazards assumption was evaluated both statistically and visually.
Random Survival Forests (RSF):
An RSF model was built to predict survival outcomes.
Permutation-based variable importance analysis was performed.
The RSF model’s concordance index (C-index) was compared to that of the Cox model. 

Installations and Dependencies

Python 3.8 or later

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, statistics, CoxPHFitter
import numpy as np
from lifelines.statistics import logrank_test
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance

Assignment Structure

Assignment-4
├── Data
│   └── RADCURE_Clinical_v04_20241219.xlsx   
├── main.ipynb                               
├── README.md     

Observations & Conclusions

Kaplan–Meier Analysis:
Significant differences in survival were observed between age groups (p-value ≈ 1.99e-36).
Cox Regression:
The model (with a C-index of 0.70) highlighted Age and Stage as significant predictors, though some proportional hazards violations were noted.
Random Survival Forests:
RSF provided a slightly higher C-index (0.72) and identified similar key predictors; however, RSF is less interpretable compared to the Cox model.
Overall, the analysis demonstrates that traditional and machine-learning methods can complement each other in survival analysis, with RSF offering improved predictive performance at the cost of interpretability.
>>>>>>> 49573f6 (Initial commit for Assignment-4)
