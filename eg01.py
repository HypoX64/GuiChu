import numpy as np
from scipy.io import wavfile
import matplotlib.pylab as plt
import os
import random
import scipy.signal
import time
import cv2
import librosa
from util import util,ffmpeg,dsp,sound,notation
from util import array_operation as arrop
from util import image_processing as impro

# hit.wav basefreq:5300 maxfreq:2000 -> as:262 HZ
# bpf:100, 20000  
# electric.wav basefreq:8000 maxfreq:9400 -> as:440 HZ
# bpf:300, 20000

# util.makedirs('./dataset/mosquito/imgs/electric')
# util.makedirs('./dataset/mosquito/imgs/hit')
# ffmpeg.video2image('./dataset/mosquito/electric.mp4', os.path.join('./dataset/mosquito/imgs/electric','%05d.jpg'))
# ffmpeg.video2image('./dataset/mosquito/hit.mp4', os.path.join('./dataset/mosquito/imgs/hit','%05d.jpg'))

notations = notation.readscore('./music/CroatianRhapsody.txt')

_,seed_voice_0 = sound.load('./dataset/mosquito/electric.wav')
seed_voice_0 = dsp.bpf(seed_voice_0, 44100, 300, 5000)

_,seed_voice_1 = sound.load('./dataset/mosquito/hit.wav')
seed_voice_1 = dsp.bpf(seed_voice_1, 44100, 20, 400)
#seed_voice_1 = sound.time_correct(seed_voice_1, out_time=0.5)

sinmusic,musicinfos = notation.notations2music(notations,mode='sin')
# print(musicinfos)
# # Generate music
music = np.zeros_like(sinmusic)
for i in range(len(musicinfos['time'])):
    for j in range(len(musicinfos['freq'][i])):
        if musicinfos['freq'][i][j]!=0:
            #print(musicinfos['freq'][i][j])
            if musicinfos['freq'][i][j]>260:
                _tone = seed_voice_0
                _tone = sound.freq_correct(_tone, srcfreq=440, dstfreq=musicinfos['freq'][i][0],alpha=1.0)
            else:
                _tone = seed_voice_1
                _tone = sound.freq_correct(_tone, srcfreq=260, dstfreq=musicinfos['freq'][i][0],alpha=1.0)
                _tone = _tone + sound.freq_correct(seed_voice_0, srcfreq=440, dstfreq=musicinfos['freq'][i][0],alpha=1.0)
            music[int(musicinfos['time'][i]*44100):int(musicinfos['time'][i]*44100)+len(_tone)] += _tone
music = music+sinmusic*0.1
music = (arrop.sigmoid(music/32768)-0.5)*65536
sound.playtest(music)
sound.write(music)




electric_imgs = []
paths = os.listdir('./dataset/mosquito/imgs/electric')
for i in range(len(paths)):
    electric_imgs.append(cv2.imread(os.path.join('./dataset/mosquito/imgs/electric',paths[i])))

hit_imgs = []
paths = os.listdir('./dataset/mosquito/imgs/hit')
for i in range(len(paths)):
    hit_imgs.append(cv2.imread(os.path.join('./dataset/mosquito/imgs/hit',paths[i])))

fps = 60
output_img_num = int(len(music)*fps/44100.0)
black_ground = np.zeros((1080, 1920, 3),dtype=np.uint8)
black_ground[:810,:1440] = cv2.resize(electric_imgs[-1],(1440,810))
for i in range(output_img_num):
    print('black_ground:  ',i,'/',output_img_num)
    if i < output_img_num - fps:
        plt.figure(figsize=(19.2,2.7))
        spectrum = dsp.signal2spectrum(music[int(i*44100/60.0):int(i*44100/60.0)+44100],512, 256, n_downsample=1, log = True, log_alpha = 0.1)
        spectrum = cv2.resize(spectrum, (192,27))
        plt.imshow(spectrum)
        plt.savefig('./tmp/spectrum_eg.jpg')
        plt.close('all')
        spectrum = cv2.imread('./tmp/spectrum_eg.jpg')
        black_ground[810:,:] = spectrum
    cv2.imwrite(os.path.join('./dataset/mosquito/imgs/output','%05d' % i+'.jpg'), black_ground)

# #Generate video

for i in range(len(musicinfos['time'])):
    print('processing:  ',i,'/',len(musicinfos['time']))
    img_cnt = int(musicinfos['time'][i]*fps)

    for j in range(len(musicinfos['freq'][i])):
        try :
            if musicinfos['freq'][i][j]!=0:
                if musicinfos['freq'][i][j]>260:
                    imgs = electric_imgs
                else:
                    imgs = hit_imgs           
                for x in range(len(imgs)):
                    img = cv2.imread(os.path.join('./dataset/mosquito/imgs/output','%05d' % (x+img_cnt)+'.jpg'))
                    if j == 0: 
                        _img = cv2.resize(imgs[x],(1440,810))
                        img[:810,:1440] = _img
                    if j == 1:
                        _img = cv2.resize(imgs[x],(480,270))
                        img[0:270,1440:1920] = _img
                    if j == 2:
                        _img = cv2.resize(imgs[x],(480,270))
                        img[270:540,1440:1920] = _img
                    if j == 3:
                        _img = cv2.resize(imgs[x],(480,270))
                        img[540:810,1440:1920] = _img

                    cv2.imwrite(os.path.join('./dataset/mosquito/imgs/output','%05d' % (x+img_cnt)+'.jpg'), img)
        except Exception as e:
            print(e)



ffmpeg.image2video(60, os.path.join('./dataset/mosquito/imgs/output','%05d.jpg'), './tmp/test_output.wav', './tmp/result.mp4')
