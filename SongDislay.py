# -*- coding: utf-8 -*-
"""
Created on Sun May 20 14:24:11 2018

@author: Jakub
"""
import librosa
import librosa.feature
import librosa.display
import matplotlib.pyplot as plt
song="f.wav"
def display_mfcc(song): #reprezentacja posczegolnej piosenki
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)
    
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()
display_mfcc(song)