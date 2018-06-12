# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:56:13 2018

@author: Jakub
"""
import keras
import keras.models
import librosa
import librosa.feature
import numpy as np
from sys import argv

def extract_features_song(f):
   y, _ =librosa.load(f)
   mfcc = librosa.feature.mfcc(y)
   mfcc /= np.amax(np.absolute(mfcc))
   return np.ndarray.flatten(mfcc)[:25000]

genres =['classical', 'country', 'disco', 'dubstep', 'jazz', 'metal', 'pop','rap', 'reagge', 'rock']
loaded_model = keras.models.load_model("genres.model")

print("Loaded model from disk")

#print(loaded_model.summary())
path = argv[1]
#print(path)
song = extract_features_song(path)

prediction = loaded_model.predict(song.reshape(1,25000))
tab = np.argmax(prediction)
#print(prediction)
print ("Prediction for %s: %s, confidence: %.2f" %(path, genres[tab], np.max(prediction)))
