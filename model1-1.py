from keras.models import Sequential, Model
from keras.layers import Input, Dense
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.layers import concatenate
import pandas as pd
import numpy as np
import string
import tensorflow as tf
import logging
from keras import optimizers

def main():
    data = pd.read_csv('../../adults/adults1.data')
    dataTest = pd.read_csv('../../adults/adults1.test')
    lr = 0.001
    epochs = 1
    decay = lr/(epochs*4)
    # momentum = .9
    momentum = 0
    sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)

    # Grab the categorical values
    X_train_cat = data[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']].values

    # Grab the continuous values
    X_train_cont = data[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']].values

    # Grab the Y
    Y_train = data[['income']].values

    # Perform one hot encoding of all categorical values
    encX = OneHotEncoder()
    encX.fit(X_train_cat)
    X_train_cat = encX.transform(X_train_cat).toarray()

    encY = OneHotEncoder()
    encY.fit(Y_train)
    Y_train = encY.transform(Y_train).toarray()

    inputs_cat = Input(shape=(102,))
    a = Dense(52,activation="sigmoid")(inputs_cat)
    a = Model(inputs=inputs_cat,outputs=a)

    inputs_cont = Input(shape=(6,))
    b = Dense(4,activation="sigmoid")(inputs_cont)
    b = Model(inputs=inputs_cont,outputs=b)

    combined = concatenate([a.output,b.output])

    z = Dense(54,activation='sigmoid')(combined)
    z = Dense(2,activation='sigmoid')(z)

    model = Model(inputs=[a.input,b.input],outputs=z)

    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

    model.fit([X_train_cat,X_train_cont],Y_train,epochs=epochs,batch_size=64)

    score = model.evaluate([X_train_cat,X_train_cont],Y_train,verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))




    # Grab the categorical values
    X_test_cat = dataTest[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']].values

    # Grab the continuous values
    X_test_cont = dataTest[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']].values

    # Grab the Y
    Y_test = dataTest[['income']].values

    # Perform one hot encoding of all categorical values
    X_test_cat = encX.transform(X_test_cat).toarray()

    Y_test = encY.transform(Y_test).toarray()

    score = model.evaluate([X_test_cat,X_test_cont],Y_test,verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    model_json = model.to_json()
    with open('model1.json','w') as json_file:
        json_file.write(model_json)
    model.save_weights('model1.h5')
    print("Saved model to disk")

if __name__ == '__main__':
    main()
