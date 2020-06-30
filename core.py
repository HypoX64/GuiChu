import numpy as np
from scipy.io import wavfile
import matplotlib.pylab as plt
import os
import random
import scipy.signal
import time
import cv2
import librosa
from util import util,ffmpeg,dsp,sound
from util import array_operation as arrop
from util import image_processing as impro


def process_video(videopath, savedir, min_interval_time=0.1, crop_mode='peak', crop_time=0.2, rate=44100, fc=[20,8000], saveimage=True):
    """
    videopath:
    savedir:
    min_interval_time:
    crop_mode: 'peak' | 'time'
    crop_time:
    rate:
    saveimage:

    return:
    video_infos :fps,endtime,height,width
    peakindexs
    bias
    syllables

    """
    util.makedirs(savedir)
    
    # process video
    video_infos = ffmpeg.get_video_infos(videopath)
    ffmpeg.video2voice(videopath, os.path.join(savedir,'video_tmp.wav'),samplingrate=44100)
    if saveimage:
        util.makedirs(os.path.join(savedir,'imgs'))
        ffmpeg.video2image(videopath,os.path.join(savedir,'imgs','%05d.png'))
    
    # process audio
    audio,syllables,features,peakindexs,bias = process_audio(os.path.join(savedir,'video_tmp.wav'), 
        savedir, min_interval_time,crop_mode, crop_time, rate, fc)

    np.save(os.path.join(savedir,'video_infos.npy'), np.array(video_infos))
    
    return audio,syllables,features,peakindexs,bias,video_infos

def process_audio(audiopath, savedir, min_interval_time=0.1, crop_mode='peak', crop_time=0.2, rate=44100, fc=[20,8000], hpss=''):
    util.makedirs(savedir)
    # to wav
    if (os.path.splitext(audiopath)[1]).lower() != '.wav':
        ffmpeg.video2voice(audiopath, os.path.join(savedir,'video_tmp.wav'),samplingrate=44100)
        audiopath = os.path.join(savedir,'video_tmp.wav')
    
    _,audio = sound.load(audiopath,ch=0)
    _audio = audio.copy()
    if hpss == 'harmonic':
        harmonic,percussive = librosa.effects.hpss(_audio)
        energy = dsp.energy(sound.filter(harmonic,fc,rate), 4410, 441, 4410)
    elif hpss == 'percussive':
        harmonic,percussive = librosa.effects.hpss(_audio)
        energy = dsp.energy(sound.filter(percussive,fc,rate), 4410, 441, 4410)
    else:
        energy = dsp.energy(sound.filter(_audio,fc,rate), 4410, 441, 4410)
    
    peakindexs = arrop.findpeak(energy,interval = int(min_interval_time*100))
    y = arrop.get_y(peakindexs, energy)
    plt.plot(energy)
    plt.scatter(peakindexs,y,c='orange')
    plt.show()

    peakindexs = peakindexs*441


    bias = []
    if crop_mode == 'peak':
        valleyindexs = arrop.findpeak(energy,interval = int(min_interval_time*100),reverse=True)*441  
        for i in range(len(peakindexs)):
            for j in range(len(valleyindexs)-1):
                if valleyindexs[j] < peakindexs[i]:
                    if valleyindexs[j+1] > peakindexs[i]:
                        left = np.clip(peakindexs[i]-valleyindexs[j],int(min_interval_time*rate*0.5),int(min_interval_time*rate*5))
                        right = np.clip(valleyindexs[j+1]-peakindexs[i],int(min_interval_time*rate*0.5),int(min_interval_time*rate*5))
                        bias.append([left,right])
    elif crop_mode == 'time':
        for i in range(len(peakindexs)):
            bias.append([int(rate*crop_time/2),int(rate*crop_time/2)])

    syllables = []
    features = []        
    for i in range(len(peakindexs)):
        syllable = audio[peakindexs[i]-bias[i][0]:peakindexs[i]+bias[i][1]]
        
        syllables.append(syllable)
        features.append(sound.freqfeatures(syllable, 44100))

    # save
    np.save(os.path.join(savedir,'peakindexs.npy'), np.array(peakindexs))
    np.save(os.path.join(savedir,'bias.npy'), np.array(bias))
    np.save(os.path.join(savedir,'syllables.npy'), np.array(syllables))
    np.save(os.path.join(savedir,'features.npy'), np.array(features))
    
    # for syllable in syllables:
    #     sound.playtest(syllable)
    
    return audio,syllables,features,peakindexs,bias

def rhythm_A_by_B(Adata,Bdata):
    """Make A has B's rhythm, but A's order does not change.
    Input:
    Adata: [Aaudio,Asyllables,Afeatures,Apeakindexs,Abias]
    Bdata; [Baudio,Bsyllables,Bfeatures,Bpeakindexs,Bbias]
    """
    Aaudio,Asyllables,Afeatures,Apeakindexs,Abias = Adata[0],Adata[1],Adata[2],Adata[3],Adata[4]
    Baudio,Bsyllables,Bfeatures,Bpeakindexs,Bbias = Bdata[0],Bdata[1],Bdata[2],Bdata[3],Bdata[4]
    new_audio = np.zeros_like(Baudio)
    print(len(Apeakindexs),len(Bpeakindexs))
    for i in range(min(len(Bsyllables),len(Asyllables))):
        
        if Bpeakindexs[i]-Abias[i][0]>0 and Bpeakindexs[i]+Abias[i][1]<len(Baudio):
            #print(i,Abias[i][0],Abias[i][1],(Bpeakindexs[i]+Abias[i][1])/44100)
            #new_audio[Bpeakindexs[i]-Abias[i][0]:Bpeakindexs[i]+Abias[i][1]] += Asyllables[i]
            src,dst = Asyllables[i],Bsyllables[i]

            src = sound.time_correct(src, dst)
            src = sound.energy_correct(src,dst,mode = 'track',alpha=0.5)

            left = int(Abias[i][0]/len(src)*len(src))
            new_audio[Bpeakindexs[i]-left:Bpeakindexs[i]-left+len(src)] += src

    new_audio = new_audio + Baudio*0.2
    sound.playtest(new_audio)


def make_B_by_A(Adata,Bdata,fc):
    """Make B by a single syllable
    Input:
    Adata: [single syllable]
    Bdata; [Baudio,Bsyllables,Bfeatures,Bpeakindexs,Bbias]
    """
    
    Baudio,Bsyllables,Bfeatures,Bpeakindexs,Bbias = Bdata[0],Bdata[1],Bdata[2],Bdata[3],Bdata[4]
    new_audio = np.zeros_like(Baudio)
    
    for i in range(len(Bsyllables)):
        
        src,dst = Adata.copy(),Bsyllables[i]
        dst = dsp.fft_filter(dst, 44100,fc)
        # src = sound.time_correct(src, dst)
        src = sound.freq_correct(src, dst, mode='normal',alpha=1)
        # src = sound.energy_correct(src,dst,mode = 'normal', alpha=0.5)
        
        # plt.plot(src)
        # plt.show()
        #src = src*np.hamming(len(src))

        new_audio[Bpeakindexs[i]-Bbias[i][0]:Bpeakindexs[i]-Bbias[i][0]+len(src)] += src
    #new_audio = dsp.fft_filter(new_audio, 44100, fc)

    plt.plot(new_audio)
    plt.show()

    new_audio = new_audio
    
    sound.playtest(new_audio)


def main():
    util.clean_tempfiles()

    #_,Aaudio = sound.load('./music/大.wav')
    Aaudio = dsp.sin(1000, 44100, 0.1)*10000
    # print(sound.basefreq(Aaudio, 44100, fc=3000))
    # xk,fft = dsp.showfreq(Aaudio,fs=44100,fc=5000)
    # plt.plot(xk,fft)
    # plt.show()

    # Aaudio = Aaudio[1500:]
    # print(Aaudio)
    # sound.playtest(Aaudio)
    Baudio,Bsyllables,Bfeatures,Bpeakindexs,Bbias = process_audio('./music/小星星.wav', './tmp/audio1', 0.07, 'time',crop_time=0.1,hpss='')
    #sound.playtest(dsp.fft_filter(Baudio,fs=44100,fc=[800,8000]))
    for syllable in Bsyllables:
        base = dsp.basefreq(syllable, fs=44100, fc=3000)
        print(base,librosa.hz_to_octs(base))
        xk,fft = dsp.showfreq(syllable,fs=44100,fc=3000)
        plt.plot(xk,fft)
        plt.show()
    #sound.playtest(Baudio)

    make_B_by_A(Aaudio,[Baudio,Bsyllables,Bfeatures,Bpeakindexs,Bbias],fc=[400,8000])


    # Aaudio,Asyllables,Afeatures,Apeakindexs,Abias,_ = process_video('../Video/素材/大威天龙！最全法海纯素材.mp4', './tmp/video1', 0.1, 'peak',0.2,fc=[20,8000],saveimage=False)
    # #Baudio,Bsyllables,Bfeatures,Bpeakindexs,Bbias,_ = process_video('../music/Astronomia.mp4', './tmp/video1', 0.1, 'peak',saveimage=False)
    # Baudio,Bsyllables,Bfeatures,Bpeakindexs,Bbias = process_audio('./music/黑人抬棺.mp3', './tmp/audio1', 0.07, 'peak')

    # rhythm_A_by_B([Aaudio,Asyllables,Afeatures,Apeakindexs,Abias],
    #     [Baudio,Bsyllables,Bfeatures,Bpeakindexs,Bbias])
if __name__ == '__main__':
    main()