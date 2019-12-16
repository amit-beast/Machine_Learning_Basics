#Data Preprocessing#
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

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
onehotencoder.fit(x)
x=onehotencoder.transform(x).toarray()
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)
