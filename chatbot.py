#Simple Neural training before making predictions
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is Ashleys ChatBot Script.

The "AshBot"

"""
#Load libraries
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tensorflow.compat.v1 as tf
import tflearn
import random
import json

with open('intents.json') as file:
    data = json.load(file)
  
#list of words, labels and docs
words = []
labels = []
docs_x = []
docs_y = []


for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
     
# Stem all of the words in the words list and remove duplicate elements 
# Figure out the vocabulary size of the model is
# How many words it has seen already     
# Important to change words to lowercase
# Take words list and make it a set - this will remove the duplicates
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

# Sort labels
labels = sorted(labels)

# Creating the training & testing output
# Nueral networks do not recognize strings so we convert to a 'bag' 
# Bag of words is "one hot encoded" 
# 
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    
    wrds = [stemmer.stem(w) for w in doc]
    
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

# Building model using tflearn (similar to tensorflow)

tf.reset_default_graph()

#Add fully connected layer to neural network which starts at the input data
#and has 8 neural networks
# Add another hidden layer with 8 neurons as well
# At two more layers which is our output layer with length = 0
# Activation = softmax: allows us to get probabilities for each output
# Softmax goes through and gives us a probability for each neuron in the layer 
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

# Train model
# DNN is a type of neural network
model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')
