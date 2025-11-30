OBESITY LEVELS BASED ON EATING HABITS AND PHYSICAL CONDITION MACHINE LEARNING FINAL PROJECT
===========================================================================================

Introduction  
This project explores obesity classification based on demographic and lifestyle factors.  
Motivation: Obesity is a global health issue, and predicting risk factors can support prevention and awareness.  
Dataset: UCI "Estimation of Obesity Levels based on Eating Habits and Physical Condition".  


Dataset Overview  
- 2,111 records, 17 variables.
- Variables include demographic (Age, Gender, Height, Weight) and lifestyle factors (diet, physical activity, habits).
- No missing values.

Methods / Approach  
Steps performed:  
- Exploratory Data Analysis (EDA)
- Statistical tests (ANOVA)
- Feature importance (Random Forest, with and without basic variables, and with class balancing)
- Multiclass Logistic Regression (evaluation with precision, recall, F1-score, confusion matrix)

Results Overview  
- Female data subset:
  - Average F1-score: 0.66
  - Best class: Obesity (0.81)
  - Worst class: Normal Weight (0.48)
  - Common confusion: Obesity ↔ Overweight (92/481 cases)

- Male data subset:
  - Average F1-score: 0.53
  - Best class: Obesity (0.63)
  - Worst class: Insufficient Weight (0.43)
  - Common confusion: Overweight ↔ Obesity (133/332 cases)

- Stable lifestyle factors:
  - Women: vegetables in meals, physical activity, number of main meals
  - Men: water intake, physical activity, number of main meals
  - Everyone: physical activity + number of meals remain consistent predictors

Difficulties  
- Dataset selection: many free datasets looked synthetic.
- Gender-specific modeling: separate analysis for women and men was required.
- Imbalance in obesity categories: solved by grouping and balancing.

Future Improvements  
- Expand dataset size for better generalization.
- Test other algorithms and apply cross-validation.
- Validate on external datasets.

How Could the Project Be Used?  
- Identify key factors influencing obesity.
- Support gender-specific health interventions.
- Improve awareness and education.
- Provide framework for future research.

Included Files  
- ReDI_Project2025_ML_Yuliia_Fomenko.pdf — final presentation
- ReDI_Project2025_ML_Yuliia_Fomenko.ipynb — Jupyter Notebook with code
- ObesityDataSet_raw_and_data_synthetic.csv — dataset
- README.md — project description

Required Libraries:  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
from scipy.stats import f_oneway  

Explanation of libraries:  
- pandas – data manipulation and analysis
- numpy – numerical computations
- matplotlib.pyplot – basic plotting
- seaborn – advanced visualizations and plot styling
- scikit-learn – train/test split, preprocessing, models (Random Forest, Logistic Regression), evaluation metrics
- scipy.stats – statistical tests (ANOVA)

How to Run:  
1. Open the notebook file: ReDI_Project2025_ML_Yuliia_Fomenko.ipynb (Jupyter Notebook format).
2. Download the required CSV file and place it in the same directory:
   - ObesityDataSet_raw_and_data_synthetic.csv
3. Run the first code cell to import the required libraries.
4. Continue running the cells to reproduce the analysis.

Resources  
- Dataset: UCI "Estimation of Obesity Levels based on Eating Habits and Physical Condition"
- Google Classroom materials (presentations, notebooks)
- Stack Overflow, W3Schools Python
- Copilot (proofreading, brainstorming, debugging, dataset search)

