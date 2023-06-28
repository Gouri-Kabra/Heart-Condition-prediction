#importing libraries 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

heart = pd.read_csv('heart.csv')

# convert categorical columns to numerical values
heart.replace({'Sex':{'M':1, 'F':2},'ChestPainType':{'ATA':1,'NAP':2, 'ASY':3, 'TA':4},'RestingECG':{'Normal':1,'ST':2, 'LVH':3},
                      'ExerciseAngina':{'Y':1, 'N':2},'ST_Slope':{'Up':1,'Flat':2, 'Down':3}},inplace=True)

heart_x = heart.drop(['HeartDisease'],axis=1) # input features
heart_y = heart['HeartDisease'] # output features

x_train,x_test,y_train,y_test=train_test_split(heart_x,heart_y,test_size=0.3,random_state=5)

gnb = GaussianNB()
gnb.fit(x_train, y_train)
 
# making predictions on the testing set
y_pred = gnb.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print('Prediction Accuracy: ', 100*acc ," %") 

# Make pickle file of our model
pickle.dump(gnb, open("model.pkl", "wb"))