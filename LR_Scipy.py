from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.cross_validation import train_test_split

#loading data using pandas creating a DataFrame
marks = pd.read_csv("ex2data1.txt",header=None) #location will vary on your device
marks['3']= 1 #adding bias term

#separating X and Y,preprocessing the data.......
Y = marks.iloc[:,-2:-1]
X = marks.iloc[:,:-2]
df = marks.iloc[:,-1:]
X  = pd.concat((X,df),axis=1)
X = X[['3',0,1]]
X.columns=['0','1','2']
Y.columns=['0']

x_train,x_test,y_train,y_test=train_test_split(X,Y.values.ravel(),test_size=0.19,random_state=1)  #using 19% for test_cases

model = LogisticRegression()  #creating an instance of LogisticRegression class
model.fit(x_train,y_train) #fitting the model

y_pred = model.predict(x_test)
accuracy = model.score(x_test,y_test)
print(accuracy)
