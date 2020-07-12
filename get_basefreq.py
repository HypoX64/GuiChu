import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pylab as plt
import random
import scipy.signal
import time
import librosa
from util import util,ffmpeg,dsp,sound,notation


dataset = './dataset/诸葛亮'
video_names = os.listdir(dataset)
video_names.sort()
util.clean_tempfiles(tmp_init=False)
util.makedirs('./tmp/voice')


for i in range(len(video_names)):
    ffmpeg.video2voice(os.path.join(dataset,video_names[i]),
        os.path.join('./tmp/voice','%03d' % i+'.wav'),
        samplingrate = 44100)
    voice = sound.load(os.path.join('./tmp/voice','%03d' % i+'.wav'))[1]
    base_freq = sound.basefreq(voice, 44100, 5000, mode = 'mean')
    print(video_names[i])
    print('basefreq:',base_freq)
    print('note:',librosa.hz_to_note(base_freq))
    f,fft = dsp.showfreq(voice, 44100, 5000)
    plt.plot(f,fft)
    plt.show()