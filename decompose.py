import os
import re
import numpy as np
import librosa
import sklearn.decomposition
import feat_extractor
import time
import pdb

# 3072 2183 202
class Logger(object):
    def __init__(self, fp):
        self.fp = fp
    def __call__(self, string, end='\n'):
        print(string)
        if self.fp is not None:
            self.fp.write('%s%s' % (string, end))

def retrieve(num, base1, base2, name1, name2, duet, stft, ins):
    first_ins = []
    second_ins = []
    for i in range(num):
        H, W, _ = sklearn.decomposition.non_negative_factorization(abs(stft[:,(i*202):((i+1)*202)].T), W=None, H=duet, n_components=600,
                                init='random', update_H=False, solver='mu',
                                beta_loss='kullback-leibler', tol=1e-4,
                                max_iter=2000, alpha=0., l1_ratio=0.,
                                regularization=None, random_state=None,
                                verbose=0, shuffle=False)
        ag = np.dot(base1.T, H[:,:300].T)
        vl = np.dot(base2.T, H[:,300:].T)
        s = (ag)/(ag+vl+1e-40)
        v = (vl)/(ag+vl+1e-40)
        one = np.multiply(s, stft[:,(i*202):((i+1)*202)])
        two = np.multiply(v, stft[:,(i*202):((i+1)*202)])
        first_ins.append(one)
        second_ins.append(two)
        
    H, W, _ = sklearn.decomposition.non_negative_factorization(abs(stft[:,((i+1)*202):].T), W=None, H=duet, n_components=600,
                                init='random', update_H=False, solver='mu',
                                beta_loss='kullback-leibler', tol=1e-4,
                                max_iter=2000, alpha=0., l1_ratio=0.,
                                regularization=None, random_state=None,
                                verbose=0, shuffle=False)
    ag = np.dot(base1.T, H[:,:300].T)
    vl = np.dot(base2.T, H[:,300:].T)
    s = (ag)/(ag+vl+1e-40)
    v = (vl)/(ag+vl+1e-40)
    one = np.multiply(s, stft[:,((i+1)*202):])
    two = np.multiply(v, stft[:,((i+1)*202):])
    first_ins.append(one)
    second_ins.append(two)
    first = first_ins[0]
    second = second_ins[0]
    for i in range(1,len(first_ins)):
        first = np.concatenate([first, first_ins[i]], axis=1)
        second = np.concatenate([second, second_ins[i]], axis=1)
    first = librosa.core.istft(first, hop_length=2183, win_length=3072)
    second = librosa.core.istft(second, hop_length=2183, win_length=3072)
    librosa.output.write_wav('./result_audio_test/' + ins[:-4] + '_seg1.wav', first.astype(np.float32), sr=44100, norm=False)
    librosa.output.write_wav('./result_audio_test/' + ins[:-4] + '_seg2.wav', second.astype(np.float32), sr=44100, norm=False)
    logger("Complete.")


if __name__ == "__main__":
    # load base
    log_fp = open('testset25_decompose_log.txt', 'w')
    logger = Logger(log_fp)
    start = time.time()
    xylophone = np.load("./bases/xylophone_base.npy")
    flute = np.load("./bases/flute_base.npy")
    violin = np.load("./bases/violin_base.npy")
    acoustic_guitar = np.load("./bases/acoustic_guitar_base.npy")
    cello = np.load("./bases/cello_base.npy")
    accordion = np.load("./bases/accordion_base.npy")
    saxophone = np.load("./bases/saxophone_base.npy")
    trumpet = np.load("./bases/trumpet_base.npy")
    '''
    xylophone = np.load("./bases/xylophone.npy").T
    flute = np.load("./bases/flute.npy").T
    violin = np.load("./bases/violin.npy").T
    acoustic_guitar = np.load("./bases/acoustic_guitar.npy").T
    cello = np.load("./bases/cello.npy").T
    accordion = np.load("./bases/accordion.npy").T
    saxophone = np.load("./bases/saxophone.npy").T
    trumpet = np.load("./bases/trumpet.npy").T
    '''
    instrument = {"xylophone":xylophone, "flute":flute, "violin":violin, "acoustic_guitar":acoustic_guitar, "cello":cello, 
                "accordion":accordion,"saxophone":saxophone, "trumpet":trumpet}

    names=['accordion','acoustic_guitar','cello','trumpet','flute','xylophone','saxophone','violin']

    AudioFile = "../testset25/gt_audio/"
    ImageFile = "../testset25/testimage/"
    filelist = os.listdir(AudioFile)
    for ins in filelist:
        #print(ins)
        AudioPath = os.path.join(AudioFile, ins)
        wave_data, sr = librosa.core.load(AudioPath, sr=44100)
        # stft = librosa.core.stft(wave_data, n_fft=4096, hop_length=2048)
        stft = librosa.core.stft(wave_data, n_fft=3072, hop_length=2183)
        # num = int(stft.shape[1]/215)
        num = int(stft.shape[1]/202)
        imdir = os.path.join(ImageFile, ins[:-4])
        imlist = os.listdir(imdir)
        imlist.sort(key=lambda x:int(x[:-4]))
        feat_extractor.load_model()
        probs_left=np.zeros([8])
        probs_right=np.zeros([8])
        if len(imlist) > 100:
            for im in imlist[40:70]:
                probs1, probs2=feat_extractor.get_CAM(imdir,'results',im)
                probs_left=probs_left+np.array(probs1)
                probs_right=probs_right+np.array(probs2)
        else:
            for im in imlist:
                probs1, probs2=feat_extractor.get_CAM(imdir,'results',im)
                probs_left=probs_left+np.array(probs1)
                probs_right=probs_right+np.array(probs2)
        #print(ins.find("violin"))
        instrument_name = [names[probs_left.argmax()], names[probs_right.argmax()]]
        instrument_base = [instrument[instrument_name[0]], instrument[instrument_name[1]]]
        #print(instrument_name)
        
        if instrument_name[0] == instrument_name[1]:
            logger(">>>>Special Case>>>>")
            if probs_left[np.argsort(probs_left)[-2]]/probs_left[np.argsort(probs_left)[-1]] > \
               probs_right[np.argsort(probs_right)[-2]]/probs_left[np.argsort(probs_right)[-1]]:
                probs_left=np.zeros([8])
                for im in imlist:
                    probs1, probs2=feat_extractor.get_CAM(imdir,'results',im)
                    probs_left=probs_left+np.array(probs1)  
                if names[probs_left.argmax()] == instrument_name[0]:
                    instrument_name = [names[np.argsort(probs_left)[-2]], names[probs_right.argmax()]]   
                    instrument_base = [instrument[instrument_name[0]], instrument[instrument_name[1]]]
                else:
                    instrument_name = [names[probs_left.argmax()], names[probs_right.argmax()]]
                    instrument_base = [instrument[instrument_name[0]], instrument[instrument_name[1]]]   
            else:
                probs_right=np.zeros([8])
                for im in imlist:
                    probs1, probs2=feat_extractor.get_CAM(imdir,'results',im)
                    probs_right=probs_right+np.array(probs2)  
                if names[probs_right.argmax()] == instrument_name[1]:
                    instrument_name = [names[probs_left.argmax()], names[np.argsort(probs_right)[-2]]]
                    instrument_base = [instrument[instrument_name[0]], instrument[instrument_name[1]]]
                else:
                    instrument_name = [names[probs_left.argmax()], names[probs_right.argmax()]]
                    instrument_base = [instrument[instrument_name[0]], instrument[instrument_name[1]]]  
        
        logger("left: " + instrument_name[0] + " right:" + instrument_name[1])
        logger("file: " + ins)
        duet = np.concatenate([instrument_base[0], instrument_base[1]])
        retrieve(num, instrument_base[0], instrument_base[1], instrument_name[0], instrument_name[1], duet, stft, ins)
    end = time.time()
    logger('totally cost: '+str(end - start))
