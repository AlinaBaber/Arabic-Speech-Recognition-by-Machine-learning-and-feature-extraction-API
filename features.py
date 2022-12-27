import librosa
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt


def sample_graph(samples,sample_rate):
  fig, ax = plt.subplots(figsize=(10,10))
  librosa.display.waveplot(samples, sr=sample_rate)
  ax.label_outer()
  ax.set(title='Data Respresentation')
  plt.show()

def MFCC_graph(samples):
  fig, ax = plt.subplots(figsize=(10,10))
  img = librosa.display.specshow(samples, x_axis='time', ax=ax)
  ax.set(title='MFCC')
  ax.label_outer()
  plt.show()

def melspectrogram_graph(data):
  fig, ax = plt.subplots(figsize=(10,10))
  S_dB = librosa.power_to_db(data, ref=np.max)
  img = librosa.display.specshow(S_dB, x_axis='time',
                          y_axis='mel', sr=16000,
                          fmax=8000, ax=ax)
  ax.set(title='Mel-frequency spectrogram')
  ax.label_outer()
  plt.show()

def poly_graph(data):
  fig, ax = plt.subplots(figsize=(10,10))
  times = librosa.times_like(data)
  ax.plot(times, data[1].T, alpha=0.8, label='Poly Feature')
  ax.legend()
  ax.label_outer()
  plt.show()

def zero_crossing_rate_graph(data):
  fig, ax = plt.subplots(figsize=(10,10))
  times = librosa.times_like(data)
  ax.plot(times, data[0],  label='zero crossing rate')
  ax.legend()
  ax.label_outer()
  plt.show()


def mfcc_feature(audio, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
 
    return mfcc   # it returns a np.array with size (40,'n') where n is the number of audio frames.

def melspectrogram_feature(audio, sample_rate):
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048)
 
    return melspectrogram   # it returns a np.array with size (128,'n') where n is the number of audio frames.

def poly_feature(audio, sample_rate):
    poly_features = librosa.feature.poly_features(y=audio, sr=sample_rate, n_fft=2048)
 
    return poly_features   # it returns a np.array with size (2,'n') where n is the number of audio frames.

def zero_crossing_rate_features(audio):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
 
    return zero_crossing_rate   # it returns a np.array with size (1,'n') where n is the number of audio frames.

