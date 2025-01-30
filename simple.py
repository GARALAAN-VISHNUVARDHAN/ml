import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data={
    'hours_studies':[1,2,3,4,5,6,7,8,9,10],
    'exam_scores':[35,40,45,50,60,65,70,75,85,90]
}
df=pd.DataFrame(data=data)
print(df)

x=df[['hours_studies']]
y=df['exam_scores']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
def predict_marks(hours):
    return model.predict([[hours]])#|y=mx+c|it is the direct operatin which the linearregression defaulty do
    

def predict_hours(marks):
    m=model.coef_[0]#|x=y-c/m|here your predicting hours using marks so we are reversing the linear regression
    c=model.intercept_
    hours=(marks-c)/m
    return hours
h1=float(input('enter number hours study'))
m1=float(input('enter number marks to get hours'))
hours_predict=predict_hours(m1)
marks_predict=predict_marks(h1)
print(f"predicted marks for{h1}hours of study:{marks_predict[0]}")
print(f"predicted hours for{m1} marks:{hours_predict}")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data = {
    'house_size': [500, 700, 900, 1100, 1500, 1800, 2100, 2500, 3000, 3500],
    'num_bedrooms': [1, 2, 2, 3, 3, 4, 4, 4, 5, 5],
    'house_age': [10, 8, 5, 4, 3, 12, 10, 7, 1, 15],
    'house_price': [150000, 175000, 200000, 225000, 300000, 350000, 400000, 450000, 500000, 600000]
}
df=pd.DataFrame(data)
print(df)
x=df[['house_size','num_bedrooms','house_age']]
y=df['house_price']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=43)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print(f"mse{mse:.2f}")
def predicthousevalue(hs,nb,ha):
    return model.predict([[hs,nb,ha]])[0]
    
hs=int(input("house size"))
nb=int(input("number of bed rooms"))
ha=int(input("house age"))
l=predicthousevalue(hs,nb,ha)
print(f"according your requirments this is the price of house{l:.2f} ")   