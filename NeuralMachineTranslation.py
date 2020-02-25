# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:56:19 2020

@author: asus
"""
"""
import the important module/library
"""
from keras.layers import (Bidirectional, Concatenate, Permute, Dot,
                          Input, LSTM, Multiply, RepeatVector, Dense, Activation, Lambda)
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from NMT_library import *
import matplotlib.pyplot as plt


#1. Dataset
m = 10000
dataset, human_vocab, machine_vocab, inv_vocab = load_dataset(m)
print(dataset[:10])


#2. Define initialization parameter
# X -> a processed version of human readable dates in training set after characters
    # is replaced by the index mapped.
#Y -> a a processed version of machine readable dates in training set after characters
    # is replaced by the index mapped.
#Xoh, Yoh -> one hot version of X, y
Tx, Ty = 30, 10
X, Y, Xoh, Yoh = preprocess_data(dataset, 
                                 human_vocab,machine_vocab,
                                 Tx, Ty)
print(Xoh.shape)


#3. Define a common layer as a global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis =-1)
densor = Dense(1, activation='relu')
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)


#4 Use the previous defined layers for one_step_attention()
def one_step_attention(a, s_prev):
    #1. For concatenating s_prev and all hidden states ('a'),
        #use repeator to repeat s_prev
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    
    #2. Connect the previous layer to the fully connected layers
    e = densor(concat)
    
    #3. Connect the previous layer to the softmax activation function
        #as we defined before in the activator variable
    alphas = activator(e)
    
    #4. For computing the context vector, we use dotor layer alphas and a as 
        #the variables
    context = dotor([alphas, a])
    
    return context

#5. Define the global layers that we shared to the our model
n_a = 64
n_s = 128
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)

#6. Now, we can create a model using bidirectional LSTM
def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Tx -> number of input 
    Ty -> number of output
    n_a -> hidden state size of the bi-LSTM
    n_s -> hidden state size of the post attention LSTM
    human_vocab_size -> size of dictionary of human_vocab
    machine_vocab_size -> size of dictionary of machine_vocab
    """
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    outputs = []
    
    #define pre-attention Bi-LSTM
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    
    #Iterate for Ty steps
    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_activation_LSTM_cell(context, initial_state
                            = [s, c])
        out = output_layer(s)
        outputs.append(out)
        
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    
    return model

model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
        
model.summary()   
   
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999,decay=0.01)  
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy']) 
        
s0 = np.zeros((m, n_s))  
c0 = np.zeros((m, n_s))  
outputs = list(Yoh.swapaxes(0,1))

model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

model.save('C:/tugas deep learning/upload github/image segmentation and clasification, medical/NMT.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('C:/tugas deep learning/upload github/image segmentation and clasification, medical/NMT.h5')

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']  
for example in EXAMPLES:  
  
    source = string_to_int(example, Tx, human_vocab)  
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)  
    prediction = model.predict([source, s0, c0])  
    prediction = np.argmax(prediction, axis = -1)  
    output = [inv_machine_vocab[int(i)] for i in prediction]  
  
print("source:", example)  
print("output:", ''.join(output))  





























    
    