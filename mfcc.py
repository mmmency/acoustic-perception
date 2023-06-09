from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import python_speech_features
import codecs
import csv
import wav
import librosa
sample_rate = 48000
def stft(s):
    full_spectrograms = []
    x1, x2 = wav.linearfilter(s)
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

#    plt.subplot(2,1,1)
#    plt.title('STFT')
#    plt.plot(x1)
#    plt.subplot(2,1,2)
    #times_L=times_L*400
#    plt.plot(times_L,time_energyL,c="r")
#    plt.xlabel('Time ')
#    plt.show()
    return(times_L,time_energyL)
'''
plt.subplot(2,1,1)
plt.plot(times_L,time_energyL,c="r")
plt.xlabel('Time [ms]')
for spec in full_spectrograms:
        plt.subplot(2,1,2)
        plt.pcolormesh(times_L, frequencies_L, np.abs(spectogram_L), cmap="gnuplot2")
        plt.ylim(0, 5000)
        plt.title("Left Channel")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.show()
        print('end')
  ######################################
        plt.pcolormesh(times_R, frequencies_R, spectogram_R, cmap="gnuplot2")
        plt.title("Right")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.show()
        ######################################
        sub_spec = spectogram_L - spectogram_R #左右声道之差
        plt.pcolormesh(times_R, frequencies_R, sub_spec, cmap="gnuplot2")
        plt.title("Subtracted")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [ms]')
        plt.show()
        ######################################
        plt.pcolormesh(spec, cmap="gnuplot2")
        plt.title("Combined")
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        plt.show()
'''
def MaxMinNormalization(x):#分段归一化
    Max = np.max(x)
    Min = np.min(x)
    Mean = np.mean(x)
    for i in range (0,len(x)):
        if x[i] > 5*Mean: #大噪声处，考虑有笔画数据的可能性，将其置为平均值
            x[i] = 3*Mean
    Max = np.max(x)
    for i in range (0,len(x)):
        x[i] = (x[i] - Min)/(Max-Min)
    return(x)

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
def mfcc(times_L,time_energyL,s):
    j=0
    i = 80
    start=[]
    y =[]
    x1, x2 = wav.linearfilter(s)
    time_energyL = MaxMinNormalization(time_energyL)
    while i < len(time_energyL)-42:#后续25ms无信号输入
        if time_energyL[i] > 0.45 and j == 0:
            start.append(i*400)
            for k in range(start[0]-11000,start[0]+25000):#k代表原始数据第k个取样点
                y.append(x1[k][0]) #y存储一帧中每个取样点对应的纵坐标分贝
            p = np.array(y)
            y =[]
            maccs = librosa.feature.mfcc(y=p,sr = 48000,n_mfcc = 16,hop_length = 380)#17000/380向上取整为45默认帧移为512
            mfcc_feat = python_speech_features.delta(maccs,1)
            mfcc_feat2=python_speech_features.delta(maccs,2)
            mfcc_feat = pinjie(mfcc_feat)
            mfcc_feat2 = pinjie(mfcc_feat2)
            maccs = pinjiearr(maccs)
            maccs = np.hstack([maccs, mfcc_feat, mfcc_feat2])

            data_write_csv('mfcc.csv',maccs)
            i+=50 #两个起始点相差大约0.5ms，60个取样点
            j+=1
        else :
            if time_energyL[i] >= 0.45 and time_energyL[i] > time_energyL[i-1]:
                start.append(i*400)
                for k in range(start[j] - 11000, start[j] + 25000):  # k代表原始数据第k个取样点
                    y.append(x1[k][0])  # y存储一帧中每个取样点对应的纵坐标分贝
                p = np.array(y)
                y = []
                maccs = librosa.feature.mfcc(y=p, sr=48000, n_mfcc=16, hop_length=380)  # 17000/380向上取整为20默认帧移为512
                mfcc_feat = python_speech_features.delta(maccs, 1)
                mfcc_feat2 = python_speech_features.delta(maccs, 2)
                mfcc_feat = pinjie(mfcc_feat)
                mfcc_feat2 = pinjie(mfcc_feat2)
                maccs = pinjiearr(maccs)
                maccs = np.hstack([maccs, mfcc_feat, mfcc_feat2])
                data_write_csv('mfcc.csv', maccs)
#                plt.title("stroke%d"%(j+1))
#                plt.plot(x1)
#                plt.xlim(start[j]-5000, start[j] + 10000)
#                plt.show()
                j=j+1
                i=i+50
        i+=1
    return (len(start),x1,x2)
