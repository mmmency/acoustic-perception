from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
# from pyAudioAnalysis import audioBasicIO
# from pyAudioAnalysis import ShortTermFeatures as sf
from segmentaxis import segment_axis
#import librosa

import scipy
#from gcc_phat import gcc_phat
import copy
#import xcorr
import filterbanks as fb
import scipy.signal as sig
import heapq
from scipy import signal
def energy(frame):
    """Computes signal energy of frame"""
    # p = 0
    # for i in frame:
    #     p += pow(i, 2) #i的二次方
    return sum(frame ** 2) / np.float64(len(frame)) #frame的平方和除以frame的长度

def fft_wav(waveData, plots=True):#画时域图
    f_array = np.fft.fft(waveData)#快速傅里叶变换
    f_abs = f_array
    axis_f = np.linspace(0, 250, np.int(len(f_array)/2))#创建等差数列
    if plots == True:
        plt.figure(dpi=100)
        plt.plot(axis_f, np.abs(f_abs[0:len(axis_f)]))
        # plt.plot(axis_f, np.abs(f_abs))
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude spectrum")
        plt.title("Tile map")
        plt.show()
    return f_abs

def filterwav(x):
    wavefft = fft_wav(x) #函数调用得到幅度谱

    step_hz = 250 / (x.size / 2)
    tab_hz = 20
    p = 3 * (x.size / 2) / 250 #步长
    savewav = []
    for i in range(int(p)):
        savewav.append(wavefft[i])
    for j in range(int(p), (len(wavefft) - int(p))):
        savewav.append(0)
    for i in range((len(wavefft) - int(p)), len(wavefft)):
        savewav.append(wavefft[i])

    axis_f = np.linspace(0, 250, np.int(len(wavefft) / 2))

    plt.figure(dpi=100)
    plt.plot(axis_f, np.abs(savewav[0:len(axis_f)]))
    # plt.plot(axis_f, np.abs(savewav))
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude spectrum")
    plt.title("Tile map after wave filtering")
    plt.show()

    i_array = np.fft.ifft(savewav)


    # plt.figure(dpi=100)
    # plt.plot(i_array.real)
    # # plt.xlabel("Time")
    # plt.ylabel("Amplitude")
    # plt.title("Single channel wavedata after wave filtering")
    # plt.show()
    return i_array.real

    # tau, cc = gcc_phat(x1, x2, fs=sample_rate)
    # print(tau, cc)

def deletequiet(file):
    sample_rate, sig = wavfile.read(file)
    print("sample_rate: %d" % sample_rate)
    N = len(sig)
    x1 = sig[:, 0]#sig所有行的第0数据左声道音频信号
    x2 = sig[:, 1]

    en = []
    window = 300
    i = 0
    endding1 = 0
    endding2 = 0

    # while i < N:
    #     if energy(x1[i:i + window]) == 0:
    #         endding1 = i + window
    #     i = i + 100
    #
    # x1 = x1[endding1:]
    # x2 = x2[endding1:]
    # plt.plot(x1)
    # plt.plot(x2)
    # plt.show()
    pp = copy.deepcopy(x1)
    pp = pp - np.mean(pp)  # 消除直流分量，均值
    p = pp / np.max(np.abs(pp))
    qq = copy.deepcopy(x2)
    qq = qq - np.mean(qq)  # 消除直流分量
    q = qq / np.max(np.abs(qq))

    # plt.plot(p)
    # plt.plot(q, c="r")
    # plt.show()

    return p,q

def devide(x, y):
    N = len(x)
    window = 6000
    pace = 300
    i = 0
    pas = 0
    p = q = 0
    cut = []
    count = 0
    last = 1
    # while i < N:
    #     if energy(x[i:i + window]) > pas:
    #         pas = energy(x[i:i + window])
    #         p = i
    #         q = p + window
    #     i = i + pace
    # re = x[p:q]
    while i < N:
        if (count >= 20000) & (last == 1):
            print(len(cut)/2)
            last = 0
        if (x[i] > 0.2) | (y[i] > 0.2):
            cut.append(i - 100)
            cut.append(i + 8000)
            i = i + 10000
            count = 0
            last = 1
            # plt.subplot(111)
            continue
        else:
            count += 1
        i = i + 1
    return cut

def devivdelib(x,y):
    N = len(x)
    window = 2000
    pace = 300
    i = 0
    pas = 0
    p = q = 0
    cut = []
    count = 0
    last = 1
    # while i < N:
    #     if energy(x[i:i + window]) > pas:
    #         pas = energy(x[i:i + window])
    #         p = i
    #         q = p + window
    #     i = i + pace
    # re = x[p:q]
    while i < N:
        if (count >= 30000) & (last == 1):
            print(len(cut) / 2)
            last = 0
        if (x[i] > 0.6) | (y[i] > 0.6):
            cut.append(i - 150)
            cut.append(i + 8500)
            i = i + 10000
            count = 0
            last = 1
            # plt.subplot(111)
            continue
        else:
            count += 1
        i = i + 1
    return cut

# def gethighenergy(x):
#

def devidehighenergy(x,y):
    N = len(x)
    window = 2000
    pace = 300
    i = 0
    pas = 0
    cut = []
    energy = []
    count = 0
    last = 1
    sample_rate = 48000
    frequencies_L, times_L, spectogram_L = signal.spectrogram(x, sample_rate, nperseg=1000,
                                                              noverlap=600, window="hamming", mode="magnitude")
    s = list(map(list, zip(*spectogram_L)))
    for p in s:
        energy.append(np.mean(p[42:106]))
    # plt.plot(energy)
    fs = 120
    fs2 = 48000
    time1 = np.arange(0, len(energy)) * (1.0 / fs)
    time2 = np.arange(0, len(x)) * (1.0 / fs2)
    plt.subplot(2, 1, 1)
    # fig, ax = plt.subplots()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.plot(time2, x, color='black')
    plt.xlabel('Time(s)')
    plt.title('Waveform')
    plt.xlim([0, 5])
    # plt.plot(time1, energy, color="black")
    plt.subplot(2,1,2)
    # fig, ax = plt.subplots()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.plot(time1, energy, color=(76 / 255, 114 / 255, 176 / 255))
    plt.xlabel('Time(s)')
    plt.title('Spectrum energy')
    plt.xlim([0, 5])
    # plt.title('After noise elimination')
    plt.show()
    i = 0
    while i < len(energy):
        if (count > 100) & (last == 1):
            print(len(cut) / 2)
            last = 0
        if energy[i] > 0.00025:
            cut.append(i*400)
            cut.append(i*400 + 8650)
            # plt.plot(x[i*400:i*400 + 8650])
            # plt.plot(y[i*400:i*400 + 8650], color = 'red')
            # plt.show()
            i += 20
            count = 0
            last = 1
        else:
            count += 1
        i = i + 1
    return cut

def sqrt_hann(M):
    return np.sqrt(np.hanning(M))


# signal: 1D array, returns a 2D complex array

def pro_signal(signal, window='hanning', frame_len=1024, overlap=512):
    if window == 'hanning':
        # w = np.hanning(frame_len)
        w = sqrt_hann(frame_len)
    else:
        w = window
    y = segment_axis(signal, frame_len, overlap=overlap, end='cut')  # use cut instead of pad
    y = w * y
    return y

def linearfilter(file):
    x1, x2 = deletequiet(file)
#    plt.plot(x1)
#    plt.plot(x2)
#    plt.show()
    fs = 48000
    N = 16  # number of channels / filters
    low_lim = 20  # centre freq. of lowest filter最低
    high_lim = fs / 2  # centre freq. of highest filter
    leny = x1.shape[0]  # filter bank length

    linear_bank = fb.Linear(leny, fs, 16, low_lim, high_lim)
    linear_bank1 = fb.Linear(leny, fs, 16, low_lim, high_lim)

    linear_bank.filters = linear_bank.filters[:, 2:3]
    linear_bank1.filters = linear_bank1.filters[:, 2:3]
    # generate subbands for signal y
    linear_bank.generate_subbands(x1)
    linear_bank1.generate_subbands(x2)

    # exclude the first (lowpass) and last (highpass) filters
    # N.B. perfect reconstruction only possible with all filters
    linear_subbands = linear_bank.subbands
    linear_subbands1 = linear_bank1.subbands

    # calculate the envelopes using the Hilbert transform
    linear_envs = np.transpose(np.absolute(sig.hilbert(np.transpose(linear_subbands))))
    linear_envs1 = np.transpose(np.absolute(sig.hilbert(np.transpose(linear_subbands1))))

    linear_subbands = linear_subbands[300: len(linear_subbands) - 300]
    linear_subbands1 = linear_subbands1[300: len(linear_subbands1) - 300]

    return linear_subbands, linear_subbands1
def get_delay(feature):
    t = feature.iloc[:, 1].values
    y = feature.iloc[:, 0].values
    delay = []
    i = 0
    count = 1
    while count in range(27):
        d = []
        for i in range(len(y)):
            if y[i] == count:
                d.append(t[i])
        delay.append(np.mean(d))
        count += 1
    return delay
def get_keyrange(t, y):
    k = []
    for p in t:
        k.append(np.abs(p - y[0]))
    re1 = heapq.nsmallest(8, k)
    best_5 = []
    for i in list(re1):
        ind = k.index(i)
        if ind not in best_5:
            best_5.append(k.index(i))
        k[k.index(i)] = -1
    return best_5

def get_effpart(y1, y2):
    return

# x1, x2 = deletequiet('ultra.wav')
# plt.plot(x1)
# plt.plot(x2)
# plt.show()
#
# cut = devide(x1, x2)
# y1 = []
# y2 = []
# i = 0
# count = 0
# while i < len(cut):
#     p = x1[cut[i]:cut[i + 1]]
#     q = x2[cut[i]:cut[i + 1]]
#     y1.append(p)
#     y2.append(q)
#     plt.plot(p)
#     plt.plot(q)
#     plt.show()
# #     plt.show()
#     i = i + 2
# def soundinten(x):
#     pi = 3.1415926
#     t = len(x) / 48000
#     en = energy(x)
#     I =




    # plt.plot(y1[i])
    # plt.plot(y2[i], color = 'red')
    # plt.show()

