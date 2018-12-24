import os
import numpy as np
import librosa
from sklearn.decomposition import NMF

instrument_tags = {"accordion":0, "acoustic_guitar":1, "cello":2, "flute":3, "saxophone":4, "trumpet":5,
                   "violin":6, "xylophone":7}

AudioPath="../dataset/audios/solo/"
instrument_list=os.listdir(AudioPath)

train_data = []
labels = []
model = NMF(n_components=25, init='random', solver='mu', beta_loss='kullback-leibler', random_state=0, max_iter=2000)

for instrument in instrument_list:
    print("Preprocess " + instrument + "....")
    instrument_data = []
    instrument_labels = []
    label = np.zeros(8,)
    label[instrument_tags[instrument]] = 1
    InsPath = os.path.join(AudioPath, instrument)
    audiolist = os.listdir(InsPath)
    for file in audiolist:
        filename = os.path.join(InsPath, file)
        print(">>>>>Preprocess " + filename + "....")
        wave_data, sr = librosa.core.load(filename, sr=44100)
        stft = librosa.core.stft(wave_data, n_fft=3072, hop_length=2183)
        sample = int(stft.shape[1]/202)-1
        for i in range(sample):
            stft1 = stft[:, i*202:(i+1)*202]
            stft2 = stft[:, (101+i*202):(101+(i+1)*202)]
            nmf1 = model.fit_transform(abs(stft1))
            nmf2 = model.fit_transform(abs(stft2))
            instrument_data.append(nmf1)
            instrument_data.append(nmf2)
            train_data.append(nmf1)
            train_data.append(nmf2)
            instrument_labels.append(label)
            instrument_labels.append(label)
            labels.append(label)
            labels.append(label)
    np.save("./base/"+instrument+"_train_data.npy", np.array(instrument_data))
    np.save("./base/"+instrument+"_label.npy", np.array(instrument_labels))

np.save("./base/train_data.npy", np.array(train_data))
np.save("./base/label.npy", np.array(labels))
