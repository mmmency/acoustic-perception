from scipy import signal
from scipy.stats import pearsonr
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import numpy as np
import codecs
import csv
import wav
import operator
import preprocess
import python_speech_features
sample_rate = 48000
flame = 760
def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
  file_csv = codecs.open(file_name,'a','utf-8')#追加
  writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
  writer.writerow(datas)
def pinjiearr(datas):
    arr =[]
    var =[]
    y = datas[:,0]  #对每一列按照行拼接，16行，30列
    arr1=np.mean([datas[0,:]])#对每一行求平均、方差
    var1=np.var([datas[0,:]])
    arr.append(arr1)
    var.append(var1)
#    print(len(datas[0]))
    for i in range (1,len(datas[0])):
#        print(len(y))
#        print(len(datas[:,i]))
        y = np.hstack([y,datas[:,i]])
    for i in range (1,16):
        arr.append(np.mean([datas[i:,]]))
        var.append(np.var([datas[i:,]]))
    y = np.hstack([y,arr])
    y = np.hstack([y,var])
    return y
def pinjie(datas):
    y = datas[:,0]  #对每一列按照行拼接，16行，30列
#    print(len(datas[0]))
    for i in range (1,len(datas[0])):
#        print(len(y))
#        print(len(datas[:,i]))
        y = np.hstack([y,datas[:,i]])
    return y
def stft(s,x1,x2):
    plt.title("linearfilter")
    plt.plot(x1)
    plt.plot(x2)
    plt.show() #滤波结果
    full_spectrograms = []
    p = np.array(list(map(list, zip(*x1))))#zip返回由元组组成的列表map(返回迭代器)
    q = np.array(list(map(list, zip(*x2))))
    frequencies_L, times_L, spectogram_L = signal.spectrogram(p, sample_rate, nperseg=1000, noverlap=600, window="hamming", mode="magnitude")
    frequencies_R, times_R, spectogram_R = signal.spectrogram(q, sample_rate, nperseg=1000,                                                                 noverlap=600, window="hamming", mode="magnitude")
    spectogram_L = spectogram_L[0]#左声道时频图，最后一轴对应时间
    spectogram_R = spectogram_R[0]
    spectrograms_combined = np.concatenate((spectogram_L, spectogram_R), axis=0)#按列拼接数组
    spectrograms_combined = spectrograms_combined.astype("float16")#规定数据类型

    #STFT转换为一维图
    full_spectrograms.append(spectrograms_combined)
    time_energyL=[None]*len(times_L) #一维数组纵坐标
    L=0.0
    for j in range(0,len(times_L)):#i是频域，j是时间采样点
        for i in range (40,105):
            if frequencies_L[i] >= 2000 or frequencies_L[i] <=5000:
                L = L+spectogram_L[i][j]
        time_energyL[j] = L
        L=0.0
    time_energyR = [None] * len(times_R)  # 一维数组纵坐标
    R = 0.0
    for j in range(0, len(times_R)):  # i是频域，j是时间采样点
        for i in range(40, 105):
            if frequencies_R[i] >= 2000 or frequencies_R[i] <= 5000:
                R = R + spectogram_R[i][j]
        time_energyR[j] = R
        R = 0.0

    plt.subplot(2,1,1)
    plt.title("STFTL&R")
    plt.plot(times_L,time_energyL,c ='b')
    plt.subplot(2,1,2)
    #times_L=times_L*4003
    plt.plot(times_R,time_energyR,c="r")
    plt.xlabel('Time ')
    plt.show()
    return(times_L,time_energyL,times_R,time_energyR)
def MaxMinNormalization(x):#分段归一化
    Max = np.max(x)
    Min = np.min(x)
    Mean = np.mean(x)
    for i in range (0,len(x)):
        if x[i] > 5*Mean: #大噪声处，考虑有笔画数据的可能性，将其置为平均值
            x[i] =3* Mean
    Max = np.max(x)
    for i in range (0,len(x)):
        x[i] = (x[i] - Min)/(Max-Min)
    return(x)
def devideL(times_L,time_energyL,s,x1):
    j=0
    i =80
    start=[]
    windows =[]
    fn = 0
    y =[]
    time_energyL = MaxMinNormalization(time_energyL)#一维数组归一
    while  i < len(time_energyL)-42:#后续25ms无信号输入
        if time_energyL[i] > 0.45 and j == 0:
            start.append(i*400)
            for k in range(start[0]-11000,start[0]+25000):#k代表原始数据第k个取样点
                y.append(x1[k][0]) #y存储一帧中每个取样点对应的纵坐标分贝
            p = np.array(y)
            y =[] #一次笔画数据
            endwindows,fn = preprocess.preprocsss(p,flame)#一个笔画中每个帧的加窗结果
            windows = endwindows
#            plt.title("stroke1")
#            plt.plot(x1)
#            plt.xlim(start[0] - 11000, start[0] + 25000)  # 每个笔画在原始信号中持续大概占17000个采样点
#            plt.show()
            i = i+50 #两个起始点相差大约0.5ms，60个取样点
            j = j+1
        else :
            if time_energyL[i] >= 0.45 and time_energyL[i] > time_energyL[i-1] :
                start.append(i*400)
                for k in range(start[j] - 11000, start[j] + 25000):  # k代表原始数据第k个取样点
                    y.append(x1[k][0])  # y存储一帧中每个取样点对应的纵坐标分贝
                p = np.array(y)
                y = []
                endwindows,fn = preprocess.preprocsss(p,flame) #一次笔画中每个帧加窗结果
                windows = np.vstack([windows,endwindows])
                j=j+1
                i=i+50
        i = i+1
    return(start,windows,time_energyL,fn)

#右声道分段获得预处理后的数据
def devideR (start,time_energyR,x2):
    time_energyR = MaxMinNormalization(time_energyR)
    y = []
    for k in range(start[0] - 11000, start[0] + 25000):
        y.append(x2[k][0])
    p = np.array(y)
    y = []
    endwindows, fn = preprocess.preprocsss(p,flame)  # 一次笔画中每个帧加窗结果
    windows = endwindows
    for i in range(1,len(start)):
        for k in range(start[i]-11000,start[i]+25000):
            y.append(x2[k][0])
        p = np.array(y)
        y =[]
        endwindows, fn = preprocess.preprocsss(p,flame)  # 一次笔画中每个帧加窗结果
        windows = np.vstack([windows, endwindows])
    return (windows,fn)



#互相关
def cross_correlation1(nL,windowsL,windowsR,fnL):
    y = []
    for i in range(0,fnL*nL):
        hfg = np.correlate(windowsL[i],windowsR[i],"full")
        max_index, max_number = max(enumerate(hfg), key=operator.itemgetter(1))#获取第一维数据，对数据编制索引，求最大值及下标
        if i % fnL != fnL-1:
            y.append(max_index)
        if i % fnL == fnL-1:
            y.append(max_index)
            data_write_csv('tdoa.csv', y)
            y = []


def cross_correlation2(nL,windowsL,windowsR,fnL):
    y = []
    for i in range(0,fnL*nL):
        lag = []
        p = []
        r = []
        lag.append(0)
        L =list(windowsL[i])
        R = list(windowsR[i])
        result = pearsonr(L,R)
        r.append(abs(result[0]))
        p.append(result[1])
        C =R.copy() #不断右移，并记录参数
        for j in range(0,fnL):
#            C.insert(0,C.pop()) #删除最后一个元素，并将其插入到C[0]
            C.insert(0,0)
            C.remove(C[len(C)-1])
            result = pearsonr(L,C)
            r.append(abs(result[0])) #存储相关系数
            p.append(result[1])
            lag.append(j+1)
        D = R.copy() #不断左移
        for k in range(0,fnL):
            D.insert(len(D),0)
            D.remove(D[0])
            result = pearsonr(L,D)
            r.append(abs(result[0]))
            p.append(result[1])
            lag.append(-(k+1))
        index = r.index(max(r))#返回最大的相关系数的索引

 #       while p[index] > 0.05: #置信度<0.05
 #           r[index] = 0
 #           index = r.index(max(r))
        if i % fnL != fnL-1:
            y.append(lag[index])
        if i % fnL == fnL-1:
            y.append(lag[index])
            data_write_csv('tdoa.csv', y)
            y = []


def xcorr(l,r):
    L_fft = fft(l)
    R_fft = np.conj(fft(r))  # R求傅里叶变换后求共轭
    f_LR = np.multiply(L_fft,R_fft)
    demon = abs(f_LR)
    f_LR = f_LR/demon
    return sum(np.abs(ifft(f_LR)))
'''    for i in range(0,len(L_fft)):
        f_LR = L_fft[i] * R_fft[i]
        denom =np.abs(f_LR)
        sum = sum+ f_LR/denom
'''



def gcc_phat(nL,windowsL,windowsR,fnL):
    y = []
    for i in range(0, fnL * nL):
        lag = []
        r = []
        lag.append(0)
        L = list(windowsL[i])
        R = list(windowsR[i])
        r.append(xcorr(L,R))
        C = R.copy()  # 不断右移，并记录参数
        for j in range(0, fnL):
            C.insert(0, 0)
            C.remove(C[len(C) - 1])
            r.append(xcorr(L,C))  # 存储相关系数
            lag.append(j + 1)
        D = R.copy()  # 不断左移
        for k in range(0, fnL):
            D.insert(len(D), 0)
            D.remove(D[0])
            r.append(xcorr(L,R))
            lag.append(-(k + 1))
        index = r.index(max(r))  # 返回最大的相关系数的索引
        if i % fnL != fnL - 1:
            y.append(lag[index])
        if i % fnL == fnL - 1:
            y.append(lag[index])
            data_write_csv('tdoagcc_phat530.csv', y)
            y = []



def tdoa (s,x1,x2):
    times_L,time_energyL,times_R,time_energyR = stft(s,x1,x2)
    startL,windowsL,time_energyL,fnL= devideL(times_L,time_energyL,s,x1) #nL笔画数,fn帧数
    windowsR,fnR = devideR(startL,time_energyR,x2)
    #startR,windowsR,time_energyR,fnR= devide(times_R,time_energyR,s)

    nL = len(startL)#笔画数
#    plt.subplot(2,1,1)
#    plt.title("nomalization L&R")
#    plt.plot(times_L, time_energyL, c="r")
#    plt.xlabel('Time ')
#    plt.subplot(2,1,2)
#    plt.plot(times_R, time_energyR, c="b")
#    plt.show()
    #cross_correlation1(nL,windowsL,windowsR,fnL)
    cross_correlation2(nL,windowsL,windowsR,fnL)
    #gcc_phat(nL,windowsL,windowsR,fnL)


