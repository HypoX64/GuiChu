import numpy as np
import scipy.fftpack
from .array_operation import *
from .dsp import *
import librosa
from scipy.io import wavfile
import os

piano = np.array([0,27.5,29.1,30.9,32.7,34.6,36.7,38.9,41.2,43.7,46.2,49.0,51.9,55.0,58.3,61.7,65.4,69.3,73.4,77.8,82.4,87.3,92.5,98.0,103.8,110.0,116.5,123.5,130.8,138.6,146.8,155.6,164.8,174.6,185.0,196.0,207.7,220.0,233.1,246.9,261.6,277.2,293.7,311.1,329.6,349.2,370.0,392.0,415.3,440.0,466.2,493.9,523.3,554.4,587.3,622.3,659.3,698.5,740.0,784.0,830.6,880.0,932.3,987.8,1047,1109,1175,1245,1319,1397,1480,1568,1661,1760,1865,1976,2093,2217,2349,2489,2637,2794,2960,3136,3322,3520,3729,3951,4186,4400])
# piano_10 = np.array([0,43.7,73.4,123.5,207.7,349.2,587.3,987.8,1661,2794,4400])
piano_10 = np.array([0,73.4,207.7,349.2,587.3,987.8,1245,1661,2093,2794,4400])


def freq_correct(src,dst,fs=44100,alpha = 0.05,fc=3000):
    src_freq = basefreq(src, 44100,3000)
    dst_freq = basefreq(dst, 44100,3000)
    offset = int((src_freq-dst_freq)/(src_freq*0.05))
    out = librosa.effects.pitch_shift(dst.astype(np.float64), 44100, n_steps=offset)
    #print('freqloss:',round((basefreq(out, 44100,3000)-basefreq(src, 44100,3000))/basefreq(src, 44100,3000),3))
    return out

def energy_correct(src,dst,alpha=1):
    src_rms = rms(src)
    dst_rms = rms(dst)
    out = dst*(src_rms/dst_rms)*alpha
    #print('energyloss:',round((rms(out)-rms(src))/rms(src),3))
    return np.clip(out,0,65536)

def freqfeatures(signal,fs):
    signal = normliaze(signal,mod = '5_95')
    signal_fft = np.abs(scipy.fftpack.fft(signal))
    length = len(signal)
    features = []
    for i in range(len(piano_10)-1):
        k1 = int(length/fs*piano_10[i])
        k2 = int(length/fs*piano_10[i+1])
        features.append(np.mean(signal_fft[k1:k2]))
    return np.array(features)


def numpy2voice(npdata):
    voice = np.zeros((len(npdata),2))
    voice[:,0] = npdata
    voice[:,1] = npdata
    return voice

def play(path):
    os.system("paplay "+path)

def playtest(npdata,freq = 44100):
    voice = numpy2voice(npdata)
    wavfile.write('./tmp/test_output.wav', freq, voice.astype(np.int16))
    play('./tmp/test_output.wav')

def write(npdata,path='./tmp/test_output.wav',freq = 44100):
    voice = numpy2voice(npdata)
    wavfile.write(path, freq, voice.astype(np.int16))


def main():
    xp = [1, 2, 3]
    fp = [3, 2, 0]
    x = [0, 1, 1.5, 2.72, 3.14]
    print(np.interp(x, xp, fp))

if __name__ == '__main__':
    main()