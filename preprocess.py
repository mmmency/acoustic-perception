import numpy as np
# 预加重就是在对信号取样以后，插入一个一阶的高通滤波器，使声门脉冲的影响减到最小
# 只剩下声道部分。也就是对语音信号进行高频提升，用一阶FIR滤波器表示
def pre_fun(x):  # 定义预加重函数
    signal_points=len(x)  # 获取语音信号的长度
    signal_points=int(signal_points)  # 把语音信号的长度转换为整型
    # s=x  # 把采样数组赋值给函数s方便下边计算
    for i in range(1, signal_points, 1):# 对采样数组进行for循环计算
        x[i] = x[i] - 0.98 * x[i - 1]  # 一阶FIR滤波器
    return x  # 返回预加重以后的采样数组


#分帧
def frame(x, lframe, mframe):  # 定义分帧函数，帧长，帧移
    length = len(x)  # 获取语音信号的长度
    fn = int(np.ceil(length/mframe))
#    fn = ((length-lframe)/mframe) +1 # 分成fn帧 ,将帧数向上取整，需要填充
#    fn = int(np.ceil(fn))
    numzero = (fn*mframe+lframe)-length
    fillzeros = np.zeros(numzero)
    fillsignal = np.concatenate((x,fillzeros))  # 获得填充后的信号
#    print(type(fillsignal))
    # 对所有帧的时间点进行抽取，得到fn*lframe长度的矩阵d
    #0到lframe步长为1 的列表，将列表行数*帧长，列数不变；
    #0到填充后的信号长度，以帧移为步长，获得长度为fn+1的列表，将列表行数*帧长后取转置
    d = np.tile(np.arange(0, lframe), (fn, 1)) + np.tile(np.arange(0, fn*mframe, mframe), (lframe, 1)).T
    # 将d转换为矩阵形式（数据类型为int类型）
    d = np.array(d, dtype=np.int32) #d为采样点坐标数组矩阵。第i行为第i帧的采样点坐标
    signal = fillsignal[d]

#    plt.title("flame")
#    x = np.arange(0,1024,1)
#    plt.plot(x,signal[0])
#    plt.show()
    return(signal, fn, lframe)#返回帧数据、帧数、填充数目


#加窗。对每一帧求窗函数
def windows(endframe,fn,lframe):
    hanwindow = np.hanning(lframe)
    window = endframe[0] * hanwindow
    endwindow = window
    for i in range(1,fn):
        hanwindow = np.hanning(lframe)
        window = endframe[i]*hanwindow
        endwindow = np.vstack([endwindow,window])
#    plt.title("window")
#    x = np.arange(0, 1024, 1)
#    plt.plot(x, endwindow[0])
#    plt.show()
    return(endwindow,fn)


#由原始信号直接得分帧加窗后的结果
def preprocsss(x,n): #得到处理好的窗函数
    x = pre_fun(x)
    endframe,fn,lframe = frame(x,n,int(n/2))
    endwindows = windows(endframe,fn,lframe)
    return(endwindows)





