# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:38:38 2018

@author: jurij
"""

#print("costam")

import librosa
import librosa.feature
import librosa.display

import glob
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

def extract_features_song(f): #normalizacja danych (-1,1) dla sieci neuronowych
   y, _ =librosa.load(f)
   mfcc = librosa.feature.mfcc(y)
   mfcc /= np.amax(np.absolute(mfcc))
   return np.ndarray.flatten(mfcc)[:25000]

def generate_features_and_labels():
    all_features = []
    all_labels = []
    genres =['classical', 'country', 'disco', 'dubstep', 'jazz', 'metal', 'pop','rap', 'reagge', 'rock']
    for genre in genres:
        sound_files = glob.glob('genres/'+genre+'/*.wav')
        print('Processing %d songs in %s genre...' %(len(sound_files), genre))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)
    #onehot encoding
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids,len(label_uniq_ids))
    return np.stack(all_features), onehot_labels


features, labels =generate_features_and_labels()

print('Features:', np.shape(features)) 
print('Labels: ', np.shape(labels))
training_split = 0.8
alldata = np.column_stack((features, labels))
np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]

#print rozmiarow tablic, zmiania rozmiaru o 10 (ze stackowania)
print('Train shape: ',np.shape(train))
print('Test shape: ',np.shape(test))
train_input = train[:,:-10]
train_labels = train[:,-10:]

test_input = test[:,:-10]
test_labels = test[:,-10:]
print(np.shape(train_input))
print(np.shape(train_labels))
    
#siec neuronowa
model = Sequential([
        Dense(20, input_dim=np.shape(train_input)[1]),
        Activation('relu'),
        Dense(10),
        Activation('softmax')
        ])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
print('dupa')

model.fit(train_input, train_labels, epochs=10, batch_size=32, validation_split=0.2)
loss, accuracy = model.evaluate(test_input, test_labels, batch_size=32)
print("Loss: %.4f, accuracy: %.4F" %(loss, accuracy))

