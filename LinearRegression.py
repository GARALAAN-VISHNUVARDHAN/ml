#using this code we predict marks using hours of study and hours of study using marks with the simple given data
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
#enter number hours study 5
#enter number marks to get hours 77
#predicted marks for5.0hours of study:58.189655172413794
#predicted hours for77.0 marks:8.073239436619719
