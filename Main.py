import glob
import shutil
from collections import Counter
import librosa.display
from pydub import AudioSegment
from pydub.silence import split_on_silence
import warnings
import joblib
from features import *
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import librosa
import copy
from speechtotext import get_large_audio_transcription

def speechmodels(sample):
    warnings.filterwarnings('ignore')
    # for Machine Learning with random Forest

    # load, extract, and visualize the features
    features = process_data_for_ML_Rf(sample)
    # prediction
   # pred_ML_RF = predict_ML_RF(features)
   # pred_ML_KNN = predict_ML_KNN(features)
    pred_ML_SVM= predict_ML_SVM(features)
   # pred_ML_Vote= predict_ML_VOTE(features)
   # Votes=pred_ML_RF,pred_ML_KNN,pred_ML_SVM,pred_ML_Vote
    return pred_ML_SVM

def wordsmodels(sample):
    warnings.filterwarnings('ignore')
    # for Machine Learning with random Forest

    # load, extract, and visualize the features
    features = process_data_for_ML_Rf(sample)
    # prediction
    pred_ML_RF = word_predict_ML_RF(features)
    pred_ML_KNN = word_predict_ML_KNN(features)
    pred_ML_SVM= word_predict_ML_SVM(features)
    pred_ML_Vote= word_predict_ML_VOTE(features)
   # Votes=pred_ML_RF,pred_ML_KNN,pred_ML_SVM,pred_ML_Vote
    return pred_ML_RF,pred_ML_KNN,pred_ML_SVM,pred_ML_Vote

# for Machine Learning on KNN with Features
def predict_ML_KNN(features):
    loaded_model = joblib.load('content/model_knn.sav')

    pred = loaded_model.predict(features.reshape(1, 6000))

    return pred
def predict_ML_SVM(features):
    loaded_model = joblib.load('content/model_svm.sav')

    pred = loaded_model.predict(features.reshape(1, 6000))

    return pred
def predict_ML_VOTE(features):
    loaded_model = joblib.load('content/model_voting.sav')

    pred = loaded_model.predict(features.reshape(1, 6000))

    return pred

# for Machine Learning on KNN with Features
def word_predict_ML_KNN(features):
    loaded_model = joblib.load('content/model_knnwords.sav')

    pred = loaded_model.predict(features.reshape(1, 6000))

    return pred
def word_predict_ML_SVM(features):
    loaded_model = joblib.load('content/model_svmwords.sav')

    pred = loaded_model.predict(features.reshape(1, 6000))

    return pred
def word_predict_ML_VOTE(features):
    loaded_model = joblib.load('content/model_votingwords.sav')

    pred = loaded_model.predict(features.reshape(1, 6000))

    return pred
# for Machine Learning on random forest with Features
def predict_ML_RF(features):
    loaded_model = joblib.load('content/model_3000.sav')

    pred = loaded_model.predict(features.reshape(1, 6000))

    return pred
# for Machine Learning on random forest with Features
def word_predict_ML_RF(features):
    loaded_model = joblib.load('content/model_3000words.sav')

    pred = loaded_model.predict(features.reshape(1, 6000))

    return pred
class feature_analysis_graphs():
    def sample_graph(self, samples, sample_rate):
        fig, ax = plt.subplots(figsize=(10, 10))
        librosa.display.waveplot(samples, sr=sample_rate)
        ax.label_outer()
        ax.set(title='Data Respresentation')
        plt.show()

    def MFCC_graph(self, samples):
        fig, ax = plt.subplots(figsize=(10, 10))
        img = librosa.display.specshow(samples, x_axis='time', ax=ax)
        ax.set(title='MFCC')
        ax.label_outer()
        plt.show()

    def melspectrogram_graph(self, data):
        fig, ax = plt.subplots(figsize=(10, 10))
        S_dB = librosa.power_to_db(data, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',
                                       y_axis='mel', sr=16000,
                                       fmax=8000, ax=ax)
        ax.set(title='Mel-frequency spectrogram')
        ax.label_outer()
        plt.show()

    def poly_graph(self, data):
        fig, ax = plt.subplots(figsize=(10, 10))
        times = librosa.times_like(data)
        ax.plot(times, data[1].T, alpha=0.8, label='Poly Feature')
        ax.legend()
        ax.label_outer()
        plt.show()

    def zero_crossing_rate_graph(self, data):
        fig, ax = plt.subplots(figsize=(10, 10))
        times = librosa.times_like(data)
        ax.plot(times, data[0], label='zero crossing rate')
        ax.legend()
        ax.label_outer()
        plt.show()


def process_data_for_ML_KNN(samples):
    sample_rate = 16000
    #graph=feature_analysis_graphs()
    #graph.sample_graph(samples, sample_rate)
    # Extract Feautures
    MFCC = mfcc_feature(samples, sample_rate)
    #graph.MFCC_graph(MFCC)
    MSS = melspectrogram_feature(samples, sample_rate)
    #graph.melspectrogram_graph(MSS)
    poly = poly_feature(samples, sample_rate)
    #graph.poly_graph(poly)
    ZCR = zero_crossing_rate_features(samples)
    #graph.zero_crossing_rate_graph(ZCR)

    # flatten an array
    MFCC = MFCC.flatten()
    MSS = MSS.flatten()
    poly = poly.flatten()
    ZCR = ZCR.flatten()

    # adding features into single array
    features = np.concatenate((MFCC, MSS, poly, ZCR))

    # padding and trimming
    max_len = 6000

    pad_width = max_len - features.shape[0]
    if pad_width > 0:
        features = np.pad(features, pad_width=((0, pad_width)), mode='constant')

    features = features[:max_len]

    return features

def chunk_process_data_for_ML_KNN(samples):
    sample_rate = 16000
    graph=feature_analysis_graphs()
    graph.sample_graph(samples, sample_rate)
    # Extract Feautures
    MFCC = mfcc_feature(samples, sample_rate)
    graph.MFCC_graph(MFCC)
    MSS = melspectrogram_feature(samples, sample_rate)
    graph.melspectrogram_graph(MSS)
    poly = poly_feature(samples, sample_rate)
    graph.poly_graph(poly)
    ZCR = zero_crossing_rate_features(samples)
    graph.zero_crossing_rate_graph(ZCR)

    # flatten an array
    MFCC = MFCC.flatten()
    MSS = MSS.flatten()
    poly = poly.flatten()
    ZCR = ZCR.flatten()

    # adding features into single array
    features = np.concatenate((MFCC, MSS, poly, ZCR))

    # padding and trimming
    max_len = 6000

    pad_width = max_len - features.shape[0]
    if pad_width > 0:
        features = np.pad(features, pad_width=((0, pad_width)), mode='constant')

    features = features[:max_len]

    return features





def process_data_for_ML_Rf(samples):
    sample_rate = 16000
    #graph = feature_analysis_graphs()
    #graph.sample_graph(samples, sample_rate)
    # Extract Feautures
    MFCC = mfcc_feature(samples, sample_rate)
    #graph.MFCC_graph(MFCC)
    MSS = melspectrogram_feature(samples, sample_rate)
    #graph.melspectrogram_graph(MSS)
    poly = poly_feature(samples, sample_rate)
    #graph.poly_graph(poly)
    ZCR = zero_crossing_rate_features(samples)
    #graph.zero_crossing_rate_graph(ZCR)

    # flatten an array
    MFCC = MFCC.flatten()
    MSS = MSS.flatten()
    poly = poly.flatten()
    ZCR = ZCR.flatten()

    # adding features into single array
    features = np.concatenate((MFCC, MSS, poly, ZCR))

    # padding and trimming
    max_len = 6000

    pad_width = max_len - features.shape[0]
    if pad_width > 0:
        features = np.pad(features, pad_width=((0, pad_width)), mode='constant')

    features = features[:max_len]

    return features


# for Deep Learningwih features


def normalize_2d(v):
    for i in range(v.shape[0]):
        norm = np.linalg.norm(v[i])
        if norm == 0:
            v[i] = v[i]
        else:
            v[i] = v[i] / norm
    return v
    # adjust target amplitude
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def makechunk(Inputfilename, sample_rate, foldername):

    #os.remove(foldername)
    sound_file = AudioSegment.from_wav(Inputfilename)
    audio_chunks = split_on_silence(sound_file, min_silence_len=40, silence_thresh=-36)
    # os.mkdir('content')
    os.mkdir(foldername)
    for i, chunk in enumerate(audio_chunks):
        out_file = foldername+"/chunk{0}.wav".format(i)
        # print("exporting", out_file)
        chunk.export(out_file, format="wav")
    sound_files = []
    chunks = os.listdir(foldername)

    for chunk in chunks:
        chunk = foldername + "/" + chunk
        # print(chunk)
        sample, sample_rate = librosa.load(chunk, sr=sample_rate)
        sound_files.append(sample)
    from pydub.silence import detect_nonsilent

    # Convert wav to audio_segment
    audio_segment = AudioSegment.from_wav(Inputfilename)

    # normalize audio_segment to -20dBFS
    normalized_sound = match_target_amplitude(audio_segment, -20.0)
    # print("length of audio_segment={} seconds".format(len(normalized_sound) / 1000))

    # # print detected non-silent chunks, which in our case would be spoken words.
    nonsilent_data = detect_nonsilent(normalized_sound, min_silence_len=40, silence_thresh=-36, seek_step=1)
    # convert ms to seconds
    # print("start,Stop")
    chunks_timestamps=list()
    for chunks in nonsilent_data:
        chunks_timestamps.append([chunk / 1000 for chunk in chunks])
        # print([chunk / 1000 for chunk in chunks])
    return chunks_timestamps



def speechrecognitiontest(Inputfilename):
   # Inputfilename = "POC/108/001/108001_01.wav"
    sample_rate = 16000
    sample, sample_rate = librosa.load(Inputfilename, sr=sample_rate)
    pred_ML_SVM= speechmodels(sample)
    # print(pred_ML_RF,pred_ML_KNN,pred_ML_SVM,pred_ML_Vote)



    actuallfilename=''
    #if a == 0:
    #    actuallfilename = "POC/108/001/"
    #elif a == 1:
    #    actuallfilename = "POC/108/002/"
    return pred_ML_SVM

def wordrecognitiontest(Inputfilename):
   # Inputfilename = "POC/108/001/108001_01.wav"
    sample_rate = 16000
    sample, sample_rate = librosa.load(Inputfilename, sr=sample_rate)
    pred_ML_RF,pred_ML_KNN,pred_ML_SVM,pred_ML_Vote = wordsmodels(sample)
    # print(pred_ML_RF,pred_ML_KNN,pred_ML_SVM,pred_ML_Vote)



    actuallfilename=''
    #if a == 0:
    #    actuallfilename = "POC/108/001/"
    #elif a == 1:
    #    actuallfilename = "POC/108/002/"
    return pred_ML_RF,pred_ML_KNN,pred_ML_SVM,pred_ML_Vote

def fill_dtw_cost_matrix(s1, s2):
    l_s_1, l_s_2 = len(s1), len(s2)
    cost_matrix = np.zeros((l_s_1 + 1, l_s_2 + 1))
    for i in range(l_s_1 + 1):
        for j in range(l_s_2 + 1):
            cost_matrix[i, j] = np.inf
    cost_matrix[0, 0] = 0

    for i in range(1, l_s_1 + 1):
        for j in range(1, l_s_2 + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            # take last min from the window
            prev_min = np.min([cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]])
            cost_matrix[i, j] = cost + prev_min
    return cost_matrix


def dtw_calculation(samplefile, targetfile, sample_rate):
    sample1, sample_rate = librosa.load(targetfile, sr=sample_rate)
    mfcc1 = librosa.feature.mfcc(sample1, sample_rate)
    sampletest, sample_rate = librosa.load(samplefile, sr=sample_rate)
    mfccTest = librosa.feature.mfcc(sampletest, sample_rate)
    #chunk_process_data_for_ML_KNN(sample1)
    #chunk_process_data_for_ML_KNN(sampletest)
    # Remove mean and normalize each column of MFCC
    def preprocess_mfcc(mfcc):
        mfcc_cp = copy.deepcopy(mfcc)
        for i in range(mfcc.shape[1]):
            mfcc_cp[:, i] = mfcc[:, i] - np.mean(mfcc[:, i])
            mfcc_cp[:, i] = mfcc_cp[:, i] / np.max(np.abs(mfcc_cp[:, i]))
        return mfcc_cp

    mfcc1 = preprocess_mfcc(mfcc1)
    a=mfcc1.shape
    mfccTest = preprocess_mfcc(mfccTest)
    # padding and trimming
    max_len = 500

    pad_width = max_len - mfcc1 .shape[1]
    if pad_width > 0:
        mfcc1 = np.pad(mfcc1 , pad_width=((1, pad_width)), mode='constant')

    mfcc1  = mfcc1 [:max_len]
    # padding and trimming
    max_len = 500

    pad_width = max_len - mfccTest.shape[1]
    if pad_width > 0:
        mfccTest = np.pad(mfccTest, pad_width=((1, pad_width)), mode='constant')

    mfccTest = mfccTest[:max_len]
    # print(mfcc1.shape)
    # print(mfccTest.shape)
    #l2_norm = lambda mfcc1, mfccTest: (mfcc1- mfccTest) ** 2
    d1, p1 = fastdtw(mfcc1, mfccTest, dist=euclidean)
    # print(d1, p1)

    return d1



# makechunk(Inputfilename, sample_rate, foldername)
def Main(Inputfilename):
    folder = 'chunks'

    try:
        for dir in os.listdir(folder):
            shutil.rmtree(os.path.join(folder, dir))
    except OSError as e:
        print("Error: %s : %s" % (folder, e.strerror))
    #Inputfilename = "ayat/ayat_001/108001_06.wav"
    sample_rate = 16000
    Predictedfoldername = "chunks/PredictedTest"
    # os.chmod(Predictedfoldername, 0o777)
    pred_ML_RF = speechrecognitiontest(Inputfilename)
    labels = os.listdir("ayat")
    print(pred_ML_RF)
    Surah_Result = labels[np.int(pred_ML_RF)]
    chunks_timestamps=makechunk(Inputfilename, sample_rate, Predictedfoldername)
    Targetfoldername = "chunks/TargetTest"
    TargetInputfileFolder = "ayat"
    SubFolder = TargetInputfileFolder + "/" + Surah_Result + "/"
    print(SubFolder)
    train_Labels = []
    # os.chmod(Targetfoldername, 0o777)
    for filename in glob.glob(os.path.join(SubFolder,"*.wav")):
        train_Labels.append(filename)
    print(train_Labels)
    makechunk(train_Labels[0], sample_rate, Targetfoldername)
    test_chunks = []

    for filename in glob.glob(os.path.join(Predictedfoldername, '*.wav')):
        test_chunks.append(filename)
    print(test_chunks)
    target_chunks = []
    for filename in glob.glob(os.path.join(Targetfoldername, '*.wav')):
        target_chunks.append(filename)
    print(target_chunks)
    status = "Processing"
    distance=[]
    word_labels = os.listdir("worddataset/")
    wordsresult=[]
    if len(test_chunks) == len(target_chunks):
        status = "You read with good normal speed."
        print(status)
        for i in range(len(test_chunks)):
           # most_occur=get_large_audio_transcription(path)
            #pred_ML_RF, pred_ML_KNN, pred_ML_SVM, pred_ML_Vote = speechrecognitiontest(test_chunks[i])
            #Result = [word_labels[np.int(pred_ML_RF)], word_labels[np.int(pred_ML_KNN)], word_labels[np.int(pred_ML_SVM)],
            #          word_labels[np.int(pred_ML_Vote)]]
            #counter = Counter(Result)
            #most_occur = counter.most_common(1)
            #print(most_occur)
         #   wordsresult.append(most_occur)
            d1=dtw_calculation(test_chunks[i], target_chunks[i], 16000)
            #chunk_comparison(test_chunks[i], target_chunks[i])
            distance.append(d1)
    elif len(test_chunks) > len(target_chunks):
        status = "You read with fast speed read it slow."
        print(status)
        for i in range(len(target_chunks)):
            if (target_chunks[i]):
                #pred_ML_RF, pred_ML_KNN, pred_ML_SVM, pred_ML_Vote = speechrecognitiontest(test_chunks[i])
                #Result = [word_labels[np.int(pred_ML_RF)], word_labels[np.int(pred_ML_KNN)],
                #          word_labels[np.int(pred_ML_SVM)],
                #          word_labels[np.int(pred_ML_Vote)]]
                #counter = Counter(Result)
                #most_occur = counter.most_common(1)
                #print(most_occur)
                #wordsresult.append(most_occur)
                d1=dtw_calculation(test_chunks[i], target_chunks[i], 16000)
                #chunk_comparison(test_chunks[i], target_chunks[i])
                distance.append(d1)
    elif len(test_chunks) < len(target_chunks):
        status = "You read with slow speed read it normal."
        print(status)
        for i in range(len(test_chunks)):
            if (target_chunks[i]):
                #pred_ML_RF, pred_ML_KNN, pred_ML_SVM, pred_ML_Vote = speechrecognitiontest(test_chunks[i])
                #Result = [word_labels[np.int(pred_ML_RF)], word_labels[np.int(pred_ML_KNN)],
                #          word_labels[np.int(pred_ML_SVM)],
                #          word_labels[np.int(pred_ML_Vote)]]
                #counter = Counter(Result)
                #most_occur = counter.most_common(1)
                #print(most_occur[1,1])
                #wordsresult.append(most_occur )
                #print(Result)
                d1=dtw_calculation(test_chunks[i], target_chunks[i], 16000)
                #chunk_comparison(test_chunks[i], target_chunks[i])
                distance.append(d1)
    most_occur = get_large_audio_transcription(Inputfilename)
    words = most_occur.split()
    return Surah_Result,words,distance,status,chunks_timestamps
    
#if __name__ == '__main__':
#    Inputfilename = "ayat/سورة الكوثر001/108001_01.wav"
#    print(Main(Inputfilename))

#    Inputfilename = "ayat/سورة الاخلاص001/112001_31.wav"
#    print(Main(Inputfilename))

#    Inputfilename = "ayat/سورة الفلق003/113003_31.wav"
#    print(Main(Inputfilename))

#    Inputfilename = "ayat/سورة الناس002/114002_31.wav"
#    print(Main(Inputfilename))

