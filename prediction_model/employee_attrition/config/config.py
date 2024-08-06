import os
import sys
import pathlib
import employee_attrition

PACKAGE_ROOT = pathlib.Path(employee_attrition.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, 'datasets')

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'


FEATURES_TO_DROP = ['JobLevel', 'PercentSalaryHike', 'YearsAtCompany', 'EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours']
FEATURES_TO_ENCODE = ['Department', 'EducationField', 'JobRole']
FEATURES_TO_MAP = ['BusinessTravel', 'Gender', 'MaritalStatus', 'OverTime']
FEATURES_TO_LOG_TRANSFORM = ['MonthlyIncome','DistanceFromHome','TotalWorkingYears','YearsInCurrentRole','YearsSinceLastPromotion']


MAPPINGS = {'BusinessTravel': {'Travel_Rarely': 1, 'Travel_Frequently':2, 'Non-Travel':0},
            'Gender': {'Male':1, 'Female':0},
            'MaritalStatus': {'Single':1, 'Married':2, 'Divorced':2},
            'OverTime': {'Yes':1, 'No':0}
            
            }


FEATURES_NAMES = ['Age', 'BusinessTravel', 'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
       'NumCompaniesWorked', 'OverTime', 'PerformanceRating',
       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'Department_Human Resources', 'Department_Research & Development',
       'Department_Sales', 'EducationField_Human Resources',
       'EducationField_Life Sciences', 'EducationField_Marketing',
       'EducationField_Medical', 'EducationField_Other',
       'EducationField_Technical Degree', 'JobRole_Healthcare Representative',
       'JobRole_Human Resources', 'JobRole_Laboratory Technician',
       'JobRole_Manager', 'JobRole_Manufacturing Director',
       'JobRole_Research Director', 'JobRole_Research Scientist',
       'JobRole_Sales Executive', 'JobRole_Sales Representative']



TARGET = 'Attrition'


MODEL_NAME = 'classification.pkl'
MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_model')

