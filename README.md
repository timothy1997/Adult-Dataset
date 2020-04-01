# Adult-Dataset
Neural Network Approach for Adult Income Classification 

This is my approach to the Adult Data Set, which can be found at: https://archive.ics.uci.edu/ml/datasets/Adult
The goal is to predict whether income exceeds $50K/yr based on census data. I used a neural network to build this model to practice this specific machine learning model. I will be continuing to update this approach as I learn more.

3/31/2020 Update (model1-2w2v.py):

I have updated this model to take advantage of word2vec google news corpus word vector model. I used the model to transform the categorical data of the model into something smarter than simply a one hot encoding transformation. However, I didn't see much improvement. I was able to achieve a test accuracy of 84% with this model, my best so far, but mostly it performed similarly to one hot encoding. I will be continuing to work on this approach to see if I can squeeze some better accuracy out.
