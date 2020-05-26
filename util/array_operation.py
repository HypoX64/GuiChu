import numpy as np

def interp(y,length):
    xp = np.linspace(0, len(y)-1,num = len(y))
    fp = y
    x = np.linspace(0, len(y)-1,num = length)
    return np.interp(x, xp, fp)


def pad(data,padding):
    pad_data = np.zeros(padding, dtype=data.dtype)
    data = np.append(data, pad_data)
    return data

def normliaze(data,mod = 'norm',sigma = 0):
    '''
    mod: norm | std | maxmin | 5_95
    '''
    if mod == 'norm':
        return (data-np.mean(data))/sigma
    elif mod == 'std':
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma
    elif mod == 'maxmin':
        return (data-np.mean(data))/sigma
    elif mod == '5_95':
        data_sort = np.sort(data)
        th5 = data_sort[int(0.05*len(data_sort))]
        th95 = data_sort[int(0.95*len(data_sort))]
        line0 = (th5+th95)/2
        return (data-line0)/((th95-th5)/2)


def diff1d(indata,stride=1,padding=1,bias=False):

    pad = np.zeros(padding)
    indata = np.append(indata, pad)
    if bias:
        if np.min(indata)<0:
            indata = indata - np.min(indata)

    outdata = np.zeros(int(len(indata)/stride)-1)
    for i in range(int(len(indata)/stride)-1):
        outdata[i]=indata[i*stride+stride]-indata[i*stride]
    return outdata


def findpeak(indata,ismax=False,interval=2):
    '''
    return:indexs
    '''
    diff = diff1d(indata)
    indexs = []
    if ismax:
        return np.array([np.argmax(indata)])

    rise = True
    if diff[0] <=0:
        rise = False
    for i in range(len(diff)):
        if rise==True and diff[i]<=0:
            index = i
            ok_flag = True
            for x in range(interval):
                if indata[np.clip(index-x,0,len(indata)-1)]>indata[index] or indata[np.clip(index+x,0,len(indata)-1)]>indata[index]:
                    ok_flag = False
            if ok_flag:
                indexs.append(index)

        if diff[i] <=0:
            rise = False
        else:
            rise = True

    return np.array(indexs)

def get_crossing(line1,line2):
    cross_pos = []
    dif = line1-line2
    flag = 1
    if dif[0]<0:
        dif = -dif        
    for i in range(int(len(dif))):
        if flag == 1:
            if dif[i] <= 0:
                cross_pos.append(i)
                flag = 0
        else:
            if dif[i] >= 0:
                cross_pos.append(i)
                flag = 1
    return cross_pos

def get_y(indexs,fun):
    y = []
    for index in indexs:
        y.append(fun[index])
    return np.array(y)

def fillnone(arr_in,flag,num = 7):
    arr = arr_in.copy()
    index = np.linspace(0,len(arr)-1,len(arr),dtype='int')
    cnt = 0
    for i in range(2,len(arr)-2):
        if arr[i] != flag:
            arr[i] = arr[i]
            if cnt != 0:
                if cnt <= num*2:
                    arr[i-cnt:round(i-cnt/2)] = arr[i-cnt-1-2]
                    arr[round(i-cnt/2):i] = arr[i+2]
                    index[i-cnt:round(i-cnt/2)] = i-cnt-1-2
                    index[round(i-cnt/2):i] = i+2
                else:
                    arr[i-cnt:i-cnt+num] = arr[i-cnt-1-2]
                    arr[i-num:i] = arr[i+2] 
                    index[i-cnt:i-cnt+num] = i-cnt-1-2
                    index[i-num:i] = i+2
                cnt = 0
        else:
            cnt += 1
    return arr,index

def crop(data,indexs,size):
    if len(data) < size:
        data = pad(data, size-len(data))

    crop_result = []
    for index in indexs:
        if index-int(size/2)<0:
            crop_result.append(data[0:size])
        elif index+int(size/2)<0:
            crop_result.append(data[len(data)-size:len(data)])
        else:
            crop_result.append(data[index-int(size/2):index+int(size/2)])
    return np.array(crop_result)

def closest(a,array):
    '''
    return:index
    '''

    corrcoefs = []
    for i in range(len(array)):
        #corrcoefs.append(np.linalg.norm(a-array[i], ord=1))
        corrcoefs.append(np.corrcoef(a,array[i])[0][1])
    return corrcoefs.index(np.max(corrcoefs))

def match(src,dst):
    '''
    return:dstindexs
    '''
    indexs = []
    for i in range(len(src)):
        indexs.append(closest(src[i], dst))
    return np.array(indexs)



def main():
    a = [0,2,4,6,8,10]
    print(interp(a, 6))
if __name__ == '__main__':
    main()