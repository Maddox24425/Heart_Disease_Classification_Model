import streamlit as st
import pickle
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import warnings
import time
warnings.filterwarnings("ignore")

st.header('Heart Disease Prediction Using Machine Learning')

data='''Heart Disease Prediction using Machine Learning
Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes.
In this project, I analyzed a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

Algorithms Used:
- Logistic Regression
- Naive Bayes
- Support Vector Machine (Linear)
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- Artificial Neural Network (1 Hidden Layer, Keras)'''

st.subheader(data)

st.image('https://tse4.mm.bing.net/th/id/OIP.7LA1z7w-drtQmnFmC0KBNAHaE7?cb=thfvnext&pid=ImgDet&w=201&h=134&c=7&o=7&rm=3')

with open('lr_model.pkl')as f:
    chatgpt=pickle.load(f)

#Load Data
df=pd.read_csv(r'D:\Data Science\Heart_Disease_Classification_Model\Source\heart.csv')

st.sidebar.header('Select Features to Predict Heart Disease')
st.sidebar.image('https://tse4.mm.bing.net/th/id/OIP.7LA1z7w-drtQmnFmC0KBNAHaE7?cb=thfvnext&pid=ImgDet&w=201&h=134&c=7&o=7&rm=3')

all_values=[]
