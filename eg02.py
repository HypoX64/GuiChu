import numpy as np
import matplotlib.pylab as plt
import os
import random
import time
import cv2
import librosa
from util import util,ffmpeg,dsp,sound,notation
from util import array_operation as arrop
from util import image_processing as impro
#HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64
'''
                                                                                                               ,
                                                             ,+nnDDDDn1+                                     n@@@1
                   +&MMn                                  1D@@@@@@@@@@@@@&n,                                1@@@@@1
                  D@@@@@,                               nM@@@@@@@@@@@@@@@@@@&+                             1@@@@@@@1
                1M@@@@@@,                             +M@@@@D,,,,,,,,,,,1@@@@@D                          +D@@@@@@@@@D+
        ,1nn,  &@@@@@@@@nnnnnnnn1,                   +@@@@@@M&&&&1 ,&&&&M@@@@@@&                 ,nD&&M@@@@@@@@@@@@@@@@@M&&Dn,
      ,M@@@@1 ,@@@@@@@@@@@@@@@@@@@D                  @@@@@@@@@&1+,  +1DM@@@@@@@@n                &@@@@@@@@@@@@@@@@@@@@@@@@@@@&
      1@@@@@1 ,@@@@@@@@@@@@@@@@@@@n                 n@@@@@@@D  +n1  D1, +M@@@@@@M                 1M@@@@@@@@@@@@@@@@@@@@@@@Mn
      1@@@@@1 ,@@@@@@@@@@@@@@@@@@M                  D@@@@@@n ,M@@D ,@@@n ,@@@@@@@                   1M@@@@@@@@@@@@@@@@@@@Mn
      1@@@@@1 ,@@@@@@@@@@@@@@@@@@+                  1@@@@@@, D@@@D ,@@@@  &@@@@@M                     D@@@@@@@@@@@@@@@@@D
      1@@@@@1 ,@@@@@@@@@@@@@@@@@D                    M@@@@@1,&@@@D ,@@@@1,M@@@@@1                      @@@@@@@@@@@@@@@@@,
      1@@@@@1 ,@@@@@@@@@@@@@@@@@,                    +@@@@@@@@@@@D ,@@@@@@@@@@@D                      +@@@@@@@@@@@@@@@@@+
      1@@@@@1 ,@@@@@@@@@@@@@@@@n                      ,M@@@@@@@@@&+n@@@@@@@@@@n                       n@@@@@@@@@@@@@@@@@D
       D@@@@1 ,@@@@@@@@@@@@@@M1                         1M@@@@@@@@@@@@@@@@@@D,                        M@@@@@MD1+1nM@@@@@@
         ,++   ++++++++++++,                              +nM@@@@@@@@@@@MD1                           nMMD1,        1DMMD
                                                              ,+1nnnn1+,
'''
#HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64HypoX64

print('Init...')
notations = notation.readscore('./music/CroatianRhapsody.txt')
dataset = './dataset/诸葛亮'
video_names = os.listdir(dataset)
video_names.sort()
util.clean_tempfiles(tmp_init=False)
util.makedirs('./tmp/voice')


seed_voices = []
seed_freqs = []
for i in range(len(video_names)):
    ffmpeg.video2voice(os.path.join(dataset,video_names[i]),
        os.path.join('./tmp/voice','%03d' % i+'.wav'),
        samplingrate = 44100)
    voice = sound.load(os.path.join('./tmp/voice','%03d' % i+'.wav'))[1]
    #voice = dsp.bpf(voice, 44100, 20, 2000)
    base_freq = sound.basefreq(voice, 44100, 4000, mode = 'mean')
    seed_voices.append(voice)
    seed_freqs.append(base_freq)

    fps,endtime,height,width = ffmpeg.get_video_infos(os.path.join(dataset,video_names[i]))
    util.makedirs(os.path.join('./tmp/video2image','%03d' % i))
    ffmpeg.video2image(os.path.join(dataset,video_names[i]), 
        os.path.join('./tmp/video2image','%03d' % i,'%05d.jpg'))


print('Generating voice...')
sinmusic,musicinfos = notation.notations2music(notations,mode='sin')

music = np.zeros_like(sinmusic)
for i in range(len(musicinfos['time'])):
    for j in range(len(musicinfos['freq'][i])):
        if musicinfos['freq'][i][j]!=0:
            diff = np.abs(librosa.hz_to_octs(seed_freqs)-librosa.hz_to_octs(musicinfos['freq'][i][j]))
            index = np.argwhere(diff == np.min(diff))[0][0]
            _tone = seed_voices[index]
            _tone = sound.freq_correct(_tone, srcfreq = seed_freqs[index], dstfreq=musicinfos['freq'][i][j],alpha=1.0)
            _tone = sound.highlight_bass(_tone, musicinfos['freq'][i][j], seed_freqs[index])
            music[int(musicinfos['time'][i]*44100):int(musicinfos['time'][i]*44100)+len(_tone)] += _tone
# music = music+sinmusic*0.1
music = dsp.bpf(music, 44100, 20, 5000)
music = (arrop.sigmoid(music/32768)-0.5)*65536
# sound.playtest(music)
sound.write(music)

# Generate video
print('Generating video...')
util.makedirs('./tmp/output_img')
showchord = True
imgs = []
resize_imgs = []
img_dirs = os.listdir('./tmp/video2image')
img_dirs.sort()
for img_dir in img_dirs:
    img_names = os.listdir(os.path.join('./tmp/video2image',img_dir))
    img_names.sort()
    _imgs = []
    _resize_imgs = []
    for img_name in img_names:
        img_path = os.path.join('./tmp/video2image',img_dir,img_name)
        _imgs.append(cv2.imread(img_path))
        _resize_imgs.append(cv2.resize(cv2.imread(img_path),(width//4,height//4)))
    imgs.append(_imgs)
    resize_imgs.append(_resize_imgs)

frame_infos = -1*np.ones((10,int((musicinfos['time'][-1]+2)*fps),2), dtype=np.int64)
outimgcnt = 0

for i in range(len(musicinfos['time'])):
    for j in range(len(musicinfos['freq'][i])):
        if musicinfos['freq'][i][j]!=0:
            diff = np.abs(librosa.hz_to_octs(seed_freqs)-librosa.hz_to_octs(musicinfos['freq'][i][j]))
            index = np.argwhere(diff == np.min(diff))[0][0]
            for x in range(len(imgs[index])):
                frame_infos[j,int(fps*musicinfos['time'][i])+x] = np.array([index,x])

for i in range(frame_infos.shape[1]):
    outimg = np.zeros((height,width,3), dtype=np.uint8)
    if frame_infos[0,i,0] != -1:
        outimg = imgs[frame_infos[0,i,0]][frame_infos[0,i,1]]
    if showchord:
        if frame_infos[1,i,0] != -1: 
            outimg[height-50*1-height//4*1:height-50*1-height//4*0,width-50*1-width//4*1:width-50*1] = resize_imgs[frame_infos[1,i,0]][frame_infos[1,i,1]]
        if frame_infos[2,i,0] != -1: 
            outimg[height-50*2-height//4*2:height-50*2-height//4*1,width-50*1-width//4*1:width-50*1] = resize_imgs[frame_infos[1,i,0]][frame_infos[1,i,1]]
        if frame_infos[3,i,0] != -1: 
            outimg[height-50*3-height//4*3:height-50*3-height//4*2,width-50*1-width//4*1:width-50*1] = resize_imgs[frame_infos[1,i,0]][frame_infos[1,i,1]]

    cv2.imwrite(os.path.join('./tmp/output_img','%05d' % i+'.jpg'),outimg)

ffmpeg.image2video(fps, 
    os.path.join('./tmp/output_img','%05d.jpg'), 
    './tmp/test_output.wav',
     './tmp/result.mp4')
