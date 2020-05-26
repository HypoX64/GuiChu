import numpy as np
from scipy.io import wavfile
import matplotlib.pylab as plt
import os
import random
import scipy.signal
import time
import cv2
from util import util,ffmpeg,dsp,sound
from util import array_operation as arrop
from util import image_processing as impro

video_dir = './dataset/duck'
music_path = './music/黑人抬棺.mp3'
Hfreq = 20
LfreqMix = 0.1
OriMix = 0.1
EnergyAlpha = 0.3
STEP = ['generate_video']  #'preprocess','matchtest','generate_video' or 'full'
FPS = 24
CorrectFreq = False
CorrectEnergy = True
ShowPeak = True
SamplingRate = 44100
IntervalTime = 0.03

#video2voice
if 'preprocess' in STEP or 'full' in STEP:
    util.clean_tempfiles(tmp_init=True)
    names = os.listdir(video_dir)
    for i,name in enumerate(names,0):
        video_path = os.path.join(video_dir,name)
        ffmpeg.video2voice(video_path, os.path.join('./tmp/video_voice',name.replace('mp4','wav')),'wav')
        
        img_dir = os.path.join('./tmp/video_imgs','%02d' % i)
        util.makedirs(img_dir)
        ffmpeg.video2image(video_path, os.path.join(img_dir,'%05d.png'))

    ffmpeg.video2voice(music_path, './tmp/music/music.wav','wav')

if 'matchtest' in STEP or 'generate_video' in STEP or 'full' in STEP:
    '''
    dst crop and get features
    '''
    print('loading...')
    names = os.listdir('./tmp/video_voice')
    names.sort()
    audios = []
    for name in names:
        path = os.path.join('./tmp/video_voice',name)
        sampling_freq,audio = wavfile.read(path)
        audios.append(audio[:,0])
    audios = np.array(audios) 
    dst_syllables = []
    dst_indexs = []
    dst_features = []
    for audio in audios:
        energy = dsp.energy(audio, 4410, 441,4410)
        indexs = arrop.findpeak(energy,ismax=True,interval = 20)
        dst_indexs.append(indexs[0])
        syllable = arrop.crop(audio,indexs*441,int(44100*0.2))[0]
        dst_syllables.append(syllable)
        dst_features.append(sound.freqfeatures(syllable, 44100))

    '''
    src crop and get features
    '''
    sampling_freq,music = wavfile.read('./tmp/music/music.wav')
    music = music[:,0]
    if 'generate_video' in STEP or 'full' in STEP:
        endtime = int(len(music)/44100)
    else:
        endtime = 20
    
    music = music[sampling_freq*0:sampling_freq*endtime]
    musicH = dsp.fft_filter(music, sampling_freq,[Hfreq,10000])
    energy = dsp.energy(musicH, 4410, 441,4410)
    src_indexs = arrop.findpeak(energy,interval = int(IntervalTime*100))
    if ShowPeak:
        y = arrop.get_y(src_indexs, energy)
        plt.plot(energy)
        plt.scatter(src_indexs,y,c='orange')
        plt.show()

    src_syllables = arrop.crop(musicH,src_indexs*441,int(44100*0.2))
    src_features = []
    for syllable in src_syllables:
        src_features.append(sound.freqfeatures(syllable, 44100))
    match_indexs = arrop.match(src_features, dst_features)

    '''
    match src and dst
    '''
    print('matching...')
    new_music = np.zeros_like(music)
    for i in range(len(src_indexs)):
        dst_index = dst_indexs[match_indexs[i]]*441
        src_index = src_indexs[i]*441
        length = len(audios[match_indexs[i]])
        left = np.clip(int(src_index-dst_index), 0, len(new_music))
        right = np.clip(int(src_index+length-dst_index), 0, len(new_music))

        this_syllable = audios[match_indexs[i]][0:right-left]
        if CorrectFreq:
            this_syllable = sound.freq_correct(src_syllables[i], this_syllable)
        if CorrectEnergy:
            this_syllable = sound.energy_correct(src_syllables[i], this_syllable,EnergyAlpha)
        new_music[left:right] = new_music[left:right]+this_syllable
    new_music = new_music+music*OriMix +LfreqMix*dsp.fft_filter(music, sampling_freq,[0,Hfreq])
    
    if 'matchtest' in STEP:
        sound.playtest(new_music)
    else:
        sound.write(new_music)

# generate video 
print('generate video...')
if 'generate_video' in STEP or 'full' in STEP:
    dst_last_frames = []
    dst_adv_frames = np.round(np.array(dst_indexs)/100*FPS).astype(np.int64)
    names = os.listdir('./tmp/video_imgs')
    for name in names:
        dst_last_frames.append(len(os.listdir(os.path.join('./tmp/video_imgs',name))))

    fill_flags = -1*np.ones(int(src_indexs[-1]/100*FPS)+12,dtype=np.int64)

    for i in range(len(src_indexs)):
        match_index = match_indexs[i]
        adv_frame = dst_adv_frames[match_indexs[i]]
        start_frame = int(np.round(src_indexs[i]/100*FPS)-adv_frame)
        last_frame = int(dst_last_frames[match_index])

        for j in range(last_frame):
            fill_flags[start_frame+j]=match_index
            img = cv2.imread(os.path.join('./tmp/video_imgs','%02d'%match_index,'%05d'%(j+1)+'.png')) 
            impro.imwrite(os.path.join('./tmp/output_imgs','%05d'%(start_frame+j)+'.jpg'), img)
      
        if i!=(len(src_indexs)-1) and start_frame+last_frame <= int(np.round(src_indexs[i+1]/100*FPS)-adv_frame):
            for j in range(int(np.round(src_indexs[i+1]/100*FPS)-adv_frame)-start_frame-last_frame+1):
               frame = start_frame+last_frame+j
               fill_flags[frame]=match_index
               img = cv2.imread(os.path.join('./tmp/video_imgs','%02d'%match_index,'%05d'%(last_frame)+'.png'))
               impro.imwrite(os.path.join('./tmp/output_imgs','%05d'%(frame)+'.jpg'), img)  
    # print(fill_flags)
    blackground = np.zeros((480,640,3), dtype=np.uint8)
    for i in range(len(fill_flags)):
        if fill_flags[i]==-1:
            impro.imwrite(os.path.join('./tmp/output_imgs','%05d'%(i)+'.jpg'), blackground)

    ffmpeg.image2video(FPS,'./tmp/output_imgs/%5d.jpg',
            './tmp/test_output.wav',
            './result.mp4')