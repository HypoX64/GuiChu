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

videopath = '../Video/素材/大威天龙！最全法海纯素材.mp4'
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
IntervalTime = 0.07

util.clean_tempfiles(tmp_init=True)




'''
---------------------------------Video Preprocess---------------------------------
'''
fps,endtime,height,width = ffmpeg.get_video_infos(videopath)
print(fps,endtime,height,width)
ffmpeg.video2voice(videopath, './tmp/video_tmp.wav',samplingrate=44100)


_,audio = sound.load('./tmp/video_tmp.wav',ch=0)

energy = dsp.energy(audio, 4410, 441, 4410)

indexs = arrop.findpeak(energy,interval = int(IntervalTime*100))
reverse_indexs = arrop.findpeak(energy,interval = int(IntervalTime*100*0.5),reverse=True)
# syllables = []
# for i in range(len(indexs)):
#     for j in range(len(reverse_indexs)-1):
#         if reverse_indexs[j] < indexs[i]:
#             if reverse_indexs[j+1] > indexs[i]:
#                 syllables.append(audio[reverse_indexs[j]*441:reverse_indexs[j+1]*441])
#                 # util.makedirs(os.path.join('./tmp/output_imgs','%05d' % i))
#                 # print(os.path.join('./tmp/output_imgs','%05d' % i,'%05d.png'),util.second2stamp(reverse_indexs[j]/100),util.second2stamp((reverse_indexs[j+1]-reverse_indexs[j])/100))
#                 # ffmpeg.video2image(videopath, 
#                 #     os.path.join('./tmp/output_imgs','%05d' % i,'%05d.png'),
#                 #     start_time=util.second2stamp(reverse_indexs[j]/100), 
#                 #     last_time=util.second2stamp((reverse_indexs[j+1]-reverse_indexs[j])/100))
# print(len(indexs),len(syllables))

syllables,endpoints = arrop.cropbyindex(audio, indexs*441, reverse_indexs*441)


# plt.plot(energy)
# y = arrop.get_y(indexs, energy)
# plt.scatter(indexs,y,c='orange')
# y = arrop.get_y(reverse_indexs, energy)
# plt.scatter(reverse_indexs,y,c='green')
# plt.show()
