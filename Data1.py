#Data Preprocessing
#Importing the libraries

import numpy as np #for importing mathematical operations
import matplotlib.pyplot as plt #for importing plots and figures
import pandas as pd #for importing files 

#Importing the dataset
dataset=pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Populatig the missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
