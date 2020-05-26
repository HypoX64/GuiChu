import os,json

# ffmpeg 3.4.6

def video2image(videopath,imagepath):
    os.system('ffmpeg -i "'+videopath+'" -f image2 '+imagepath)

def video2voice(videopath,voicepath,_format='mp3'):
    os.system('ffmpeg -i "'+videopath+'" -f '+_format+' '+voicepath)

def image2video(fps,imagepath,voicepath,videopath):
    os.system('ffmpeg -y -r '+str(fps)+' -i '+imagepath+' -vcodec libx264 '+'./tmp/video_tmp.mp4')
    #os.system('ffmpeg -f image2 -i '+imagepath+' -vcodec libx264 -r '+str(fps)+' ./tmp/video_tmp.mp4')
    os.system('ffmpeg -i ./tmp/video_tmp.mp4 -i "'+voicepath+'" -vcodec copy -acodec libmp3lame '+videopath)

def get_video_infos(videopath):
    cmd_str =  'ffprobe -v quiet -print_format json -show_format -show_streams -i "' + videopath + '"'  
    #out_string = os.popen(cmd_str).read()
    #For chinese path
    #https://blog.csdn.net/weixin_43903378/article/details/91979025
    stream = os.popen(cmd_str)._stream
    out_string = stream.buffer.read().decode(encoding='utf-8')

    infos = json.loads(out_string)
    try:
        fps = eval(infos['streams'][0]['avg_frame_rate'])
        endtime = float(infos['format']['duration'])
        width = int(infos['streams'][0]['width'])
        height = int(infos['streams'][0]['height'])
    except Exception as e:
        fps = eval(infos['streams'][1]['r_frame_rate'])
        endtime = float(infos['format']['duration'])
        width = int(infos['streams'][1]['width'])
        height = int(infos['streams'][1]['height'])

    return fps,endtime,width,height

def cut_video(in_path,start_time,last_time,out_path,vcodec='h265'):
    if vcodec == 'copy':
        os.system('ffmpeg -ss '+start_time+' -t '+last_time+' -i "'+in_path+'" -vcodec copy -acodec copy '+out_path)
    elif vcodec == 'h264':    
        os.system('ffmpeg -ss '+start_time+' -t '+last_time+' -i "'+in_path+'" -vcodec libx264 -b 12M '+out_path)
    elif vcodec == 'h265':
        os.system('ffmpeg -ss '+start_time+' -t '+last_time+' -i "'+in_path+'" -vcodec libx265 -b 12M '+out_path)

def continuous_screenshot(videopath,savedir,fps):
    '''
    videopath: input video path
    savedir:   images will save here
    fps:       save how many images per second
    '''
    videoname = os.path.splitext(os.path.basename(videopath))[0]
    os.system('ffmpeg -i "'+videopath+'" -vf fps='+str(fps)+' '+savedir+'/'+videoname+'_%05d.jpg')
