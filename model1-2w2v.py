import gensim, logging
import pandas as pd
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from keras.optimizers import Adam
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
import pdb

def wordLookup(word):
    if word == 'holand':
        return 'holland'
    elif word == '11th':
        return 'eleventh'
    elif word == '12th':
        return 'twelfth'
    elif word == '10th':
        return 'tenth'
    elif word == 'inspct':
        return 'inspect'
    elif word == 'amer':
        return 'american'
    elif word == 'cambodia':
        return 'asia'
    elif word == 'honduras' or word == 'nicaragua' or word == 'salvador' or word == 'trinadad' or word == 'tobago' or word == 'guatemala':
        return 'americas'
    elif word == 'yugoslavia':
        return 'europe'
    elif word == '?':
        return 'unknown'
    else:
        return word

def transform(input_list,model):
    data = [line.replace(' ','').replace('-',' ').lower() for line in input_list]

    newList = []
    for input in data:
            if input == 'trinadad&tobago':
                inputs = input.split('&')
            elif input == 'assoc acdm':
                inputs = ['associate']
            elif input == 'outlying us(guam usvi etc)':
                inputs = ['outlying','american','guam']
            else:
                inputs = input.split()

            # We will be processing an array called inputs, which is one piece of data (for example, ['state', 'gov'])
            googInputs = []
            for i in range(0,len(inputs)):
                try:
                    googInputs.append(model[wordLookup(inputs[i])])
                except:
                    print(inputs[i] + ' didn\'t work ************')
            newList.append(sum(googInputs))
            newList[-1] = np.reshape(newList[-1],(300,))

    newList = np.concatenate((newList[0],newList[1],newList[2],newList[3],newList[4],newList[5],newList[6],newList[7]))
    return newList

def main():
    absolutePath = 'C:/Users/timsh/OneDrive/Documents/directory/word2vec/GoogleNews-vectors-negative300.bin'
    trainData = 'C:/Users/timsh/OneDrive/Documents/directory/adults/adults1.data'
    testData = 'C:/Users/timsh/OneDrive/Documents/directory/adults/adults1.test'

    GoogModel = KeyedVectors.load_word2vec_format(absolutePath,binary=True)
    data = pd.read_csv(trainData)
    dataTest = pd.read_csv(testData)

    lr = 0.001
    epochs = 20
    decay = lr/(epochs*4)
    momentum = 0
    # sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

    # Grab the categorical values
    X_train_cat = data[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']].values

    newTrain = []
    for line in X_train_cat:    # here we will transform each line
        newTrain.append(transform(line,GoogModel))
    X_train_cat = np.asarray(newTrain)

    # Grab the continuous values
    X_train_cont = data[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']].values

    # Grab the Y
    Y_train = data[['income']].values

    # I will still be performing one hot encoding of Y variables, I haven't decided if I want to use word2vec on that yet
    encY = OneHotEncoder()
    encY.fit(Y_train)
    Y_train = encY.transform(Y_train).toarray()


    # Here is the model for the categorical data
    inputs_cat = Input(shape=(2400,))
    a = Dense(52,activation="sigmoid")(inputs_cat)
    # a = Dense(1227,activation="sigmoid")(inputs_cat)
    a = Model(inputs=inputs_cat,outputs=a)


    # Here is the model for the continuous data
    inputs_cont = Input(shape=(6,))
    b = Dense(4,activation="sigmoid")(inputs_cont)
    b = Model(inputs=inputs_cont,outputs=b)

    # Here is where we combine
    combined = concatenate([a.output,b.output])

    # And we continuing computing both from here...
    z = Dense(54,activation='sigmoid')(combined)
    # z = Dense(600,activation='sigmoid')(combined)
    z = Dense(2,activation='sigmoid')(z)

    model = Model(inputs=[a.input,b.input],outputs=z)

    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

    model.fit([X_train_cat,X_train_cont],Y_train,epochs=epochs,batch_size=128)

    score = model.evaluate([X_train_cat,X_train_cont],Y_train,verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    # # Down here we test our model

    # Grab the categorical values
    X_test_cat = dataTest[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']].values

    # Grab the continuous values
    X_test_cont = dataTest[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']].values

    # Grab the Y
    Y_test = dataTest[['income']].values

    # Perform one hot encoding of all categorical values
    newTrain = []
    for line in X_test_cat:    # here we will transform each line
        newTrain.append(transform(line,GoogModel))
    X_test_cat = np.asarray(newTrain)

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
