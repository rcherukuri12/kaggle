# titanic_predictor.py
# this is a fun example...to see how much info. we can still retain 
# if we binarize all features.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/raja/data/titanic"

train_data = pd.read_csv(path + "/train.csv")
test_data = pd.read_csv(path + "/test.csv")
target = 'Survived'
print(train_data.columns)

# first let us drop the columns that we assume don't mean much.
cols = ['Name','Ticket','Cabin']
train_data = train_data.drop(cols,axis=1)
test_data = test_data.drop(cols,axis=1)

print(train_data.columns)

# now let us do one-hot encoding for Pclass, Sex, Embarked and then drop them
def do_one_hot(tdata,cols):
    dummies = []
    for col in cols:
        dummies.append(pd.get_dummies(tdata[col]))
    # arrange all the three columns dummies side by side
    titanic_dummies = pd.concat(dummies, axis=1)
    tdata = pd.concat((tdata,titanic_dummies),axis=1)
    tdata = tdata.drop(cols,axis=1)
    return tdata
cols = ['Pclass','Sex','Embarked']
train_data = do_one_hot(train_data,cols)
test_data = do_one_hot(test_data,cols)
# drop all males 
train_data = train_data.drop("male",axis=1)
test_data = test_data.drop("male",axis=1)

# let us switch ['Pclass','Sex','Embarked'] to category encoding.
#from category_encoders import OrdinalEncoder 


# just interpolate training data
train_data['Age'] = train_data['Age'].interpolate()
train_data['Fare'] = train_data['Fare'].interpolate()

# next Age and Fare can be made into intervals and one-hot encoded.
##  deal with fare, low price, high price, other
train_data['Fare'] = pd.cut(train_data['Fare'], [-1,20,80,1000], labels=['l','m','h'])
test_data['Fare'] = pd.cut(test_data['Fare'], [-1,20,80,1000], labels=['l','m','h'])
cols2 = ['Fare']
train_data = do_one_hot(train_data,cols2)
test_data = do_one_hot(test_data,cols2)

train_data['Age'] = pd.cut(train_data['Age'],[-1,10,20,30,40,50,60,70,80,90,100])
test_data['Age'] = pd.cut(test_data['Age'],[-1,10,20,30,40,50,60,70,80,90,100])
cols3 = ['Age']
train_data = do_one_hot(train_data,cols3)
test_data = do_one_hot(test_data,cols3)

# Parch - is family relationships it can be binarized as well
train_data['Parch'] = pd.cut(train_data['Parch'], [0,1,2,3,4,5,6], labels=['mf','ff','df','sf','ssf','sdf'])
test_data['Parch'] = pd.cut(test_data['Parch'], [0,1,2,3,4,5,6], labels=['mf','ff','df','sf','ssf','sdf'])
cols4 = ['Parch']
train_data = do_one_hot(train_data,cols4)
test_data = do_one_hot(test_data,cols4)

# # binarize cabin data as well
# train_data['Cabin'] = train_data['Cabin'].apply(lambda k: str(k)[0])
# test_data['Cabin'] = test_data['Cabin'].apply(lambda k: str(k)[0])
# # find out un-common value
# cols_d = np.setxor1d(train_data['Cabin'].unique(),test_data['Cabin'].unique())
# cols5 = ['Cabin']
# train_data = do_one_hot(train_data,cols5)
# test_data = do_one_hot(test_data,cols5)
# # drop the cabin type column that is extra for training 
# train_data = train_data.drop(cols_d,axis=1)


# now we have call columns in binary format ...they all are binary encoded/binned...ripe for keras
#Now we convert our dataframe from pandas to numpy and we assign input and output

Xtrain = train_data.values
Xtrain = np.delete(Xtrain,1,axis=1) # drop the 'Survived' data
Xtrain = np.delete(Xtrain,0,axis=1) # drop the 'PassengerId' data
Ytrain = train_data['Survived'].values
Ytrain = pd.get_dummies(Ytrain).astype('float32').values 


# for test data separate out 'PassengerId' so that you can use later.
Xtest_ids = test_data['PassengerId'].values
Xtest_org  = test_data.values

Xtest = np.delete(Xtest_org,0,axis=1) # drop the 'PassengerId' data

# Now everything is ready ....
from tensorflow import keras
from tensorflow.keras import layers , callbacks
model = keras.Sequential([
    layers.Dense(Xtrain.shape[1], activation="relu"),
    layers.Dense(Xtrain.shape[1]/4, activation="relu"),
    layers.Dense(Xtrain.shape[1]/8, activation="relu"),
    layers.Dense(2, activation="softmax")
    ])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
model.fit(Xtrain, Ytrain, epochs=1000, batch_size=16,callbacks=[callback])
#model.fit(Xtrain, Ytrain, epochs=5000, batch_size=32)

predictions = model.predict(Xtest)
# convert predictions into 
y_score = np.argmax(predictions,axis=1)

output = np.column_stack((Xtest_org[:,0],y_score))
df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
df_results.to_csv('titanic_results.csv',index=False)
