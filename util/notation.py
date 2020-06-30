import os
import sys
import random
import time
import numpy as np
import matplotlib.pylab as plt
import librosa

from . import util,ffmpeg,dsp,sound
from . import array_operation as arrop


"""音调对应的频率
* C大调中就是1=C中C唱做do,而在1=D中D要唱作do
* C D E F G A B -> do re mi fa so la xi
* 暂时不支持C# D# F# A#
"""
NOTES = ['C','D','E','F','G','A','B']
base_pitch = [16.352,18.354,20.602,21.827,24.500,27.500,30.868]
#Frequency in hertz (semitones above or below middle C)
OPS = np.zeros(7*10) 
for i in range(10):
    for j in range(7):
        OPS[i*7+j] = base_pitch[j]*(2**i)

def getfreq(note,PN):
    if note[0] == '-':
        return 0,1
    index = 4*7-1 + NOTES.index(PN) + int(note[0]) #1 = C or D.....
    if '+' in note:
        index += (len(note)-1)*7
    elif '-' in note:
        index -= (len(note)-1)*7
    freq = OPS[index]
    timescale = np.clip(1/(1.08**(index-33)),0,4.0)
    return freq,timescale

def wave(f, fs, time, mode='sin'):
    length = int(fs*time)
    signal = dsp.wave(f, fs, time, mode)
    weight = np.zeros(length)
    weight[0:length//10] = np.hanning(length//10*2)[0:length//10]
    weight[length//10:] = np.hanning(int(length*0.9*2))[-(length-length//10):]
    return signal*weight


"""拍子强度
strengths = {'2/4':['++',''],
             '3/4':['++','',''],
             '4/4':['++','','+',''],
             '3/8':['++','',''],
             '6/8':['++','','','+','','']}
"""

def getstrength(i,BN,x):
    if int(BN[0]) == 2:
        if i%2 == 0:
            x = x*1.25
    elif int(BN[0]) == 3:
        if i%3 == 0:
            x = x*1.25
    elif int(BN[0]) == 4:
        if i%4 == 0:
            x = x*1.25
        elif i%4 == 2:
            x = x*1.125
    return x

def readscore(path):
    notations = {}
    notations['data'] =[]
    for i,line in enumerate(open(path),0):
        line = line.strip('\n')
        if i==0:
            notations['PN'] = line
        elif i == 1:
            notations['BN'] = line
        elif i == 2:
            notations['BPM'] = float(line)
        elif i == 3:
            notations['PNLT'] = float(line)
        else:
            if len(line)>2:
                if line[0] != '#':
                    beat = line.split(';')
                    part = []
                    for i in range(len(beat)):
                        part.append(beat[i].split('|'))
                    notations['data'].append(part)
    return notations


def notations2music(notations, mode = 'sin',  isplot = False):
    BPM = notations['BPM']
    BN = notations['BN']
    PNLT = notations['PNLT']
    interval = 60.0/BPM 
    fs = 44100
    time = 0
    musicinfos = {'time':[],'last':[],'freq':[],'note':[]}

    music = np.zeros(int(fs*(len(notations['data'])+8)*interval))

    for section in range(len(notations['data'])):
        for beat in range(len(notations['data'][section])):
            _music = np.zeros(int(fs*PNLT*4))
            lasts = [];freqs = [];octs =[]
            
            for part in range(len(notations['data'][section][beat])):
                _note = notations['data'][section][beat][part].split(',')[0]
                freq,timescale = getfreq(_note,notations['PN'])
                
                lasts.append(PNLT*timescale)
                freqs.append(freq)
                
                if freq != 0:
                    octs.append(librosa.hz_to_note(freq))
                    _music[:int(PNLT*timescale*fs)] += wave(freq, fs, PNLT*timescale, mode = mode)
            music[int(time*fs):int(time*fs)+int(PNLT*fs*4)] += _music
            
            musicinfos['time'].append(time)
            musicinfos['last'].append(lasts)
            musicinfos['freq'].append(freqs)
            musicinfos['note'].append(octs)

            time += float(notations['data'][section][beat][0].split(',')[1])*interval

    return (arrop.sigmoid(music)-0.5)*65536,musicinfos

if __name__ == '__main__':
    main()

def main():
    notations = readscore('./music/SchoolBell.txt')
    print(notations)
    print(len(notations['data']))
    # sin triangle square
    music = notations2music(notations,mode='sin',isplot=False)
    sound.playtest(music)

# import threading
# t=threading.Thread(target=sound.playtest,args=(music,))
# t.start()
#music = notations2music(notations,mode='sin',isplot=True)
