import mfcc
import tdoa
import operator
import modlewrite
import numpy as np
from sklearn.metrics import classification_report
from functools import  reduce
num = 95
#结果转化
def category(y_hat):
    group = 0
    count = [0] * 7
    for j in range(0, len(y_hat)):
        for i in range(0, 7):
            if int(y_hat[j]) == i:
                count[i - 1] = count[i - 1] + 1
    for i in range(0, 7):  # 待识别笔画的类别数
        if count[i] != 0:
            group += 1
    t = 0
    count.append(len(y_hat))
    count.append(group)
    count = list(map(str, count))
    return(count)

def arrchange(arr):
    arr = reduce(operator.add,arr)
    return arr

def len0(x):
    sum = 0
    for i in range(0,len(x)):
        if x[i] !='0':
            sum = sum +1
    return(sum)

def contrast(y_hat,x,y,word):
    flag =0
    num_x =len0(x)
    num_y_hat = len0(y_hat)
    if num_y_hat != num_x:
        return (word)
    else:
        for i in range(0,num_x):
            if x[i]== y_hat[i] :
                flag=flag+1
        n = num_y_hat - flag
        word[n].append(y)
    return(word)


def compare(result,x,y):
    word =[]
    flag = 0 #置信度
    if operator.eq(result, x) == True:
        word.append(y)
    for k in range(0, 7):
        if int(x[7]) == len(result) and int(result[k] == x[k]) and int(result[k]) != 0:  # 相同的笔画类别
            flag = flag + 1
    if int(x[7]) == len(result) and (int(result[8]) == int(x[8]) - 1) and flag == int(x[8]) - 2:  # 笔画数目相同但有一个笔画类别识别错误
        word.append(y)
    return(word)
'''
def main():
    reader_mfcc = modlewrite.data_read_csv('pen2mfcc.csv')  # 文件中是待识别汉字的特征值
    reader_mfcc_tdoa = modlewrite.data_read_csv('pen2_tdoa760.csv')
    y_hat_mfcc = []

    X_mfcc= modlewrite.feature_read(reader_mfcc, 210, num * 16 * 3 + 16 * 2)  # 读取特征数据
    X_tdoa = modlewrite.feature_read(reader_mfcc_tdoa, 210, num)

    X = np.hstack([X_mfcc, X_tdoa])  # 时频域特征组合

    y_hat_mfcc =  modlewrite.tree_read(X_mfcc)
    y_hat_tdoa = modlewrite.tree_tdoa_read(X_tdoa)
    y_hat_mfcc_tdoa =modlewrite.tree_mfcc_tdoa_read(X)
    print("classification report:")
    target_names = ['1', '2', '3', '4', '5', '6', '7']
    y_test = ['1']*30+['2']*30+['3']*30+['4']*30+['5']*30+['6']*30+['7']*30
    print('时频域特征组合识别笔画结果：')
    print(classification_report(y_test, y_hat_mfcc_tdoa, target_names=target_names))
    print('仅频域特征识别笔画结果：')
    print(classification_report(y_test, y_hat_mfcc, target_names=target_names))
    print('仅时域特征识别笔画结果：')
    print(classification_report(y_test, y_hat_tdoa, target_names=target_names))




if __name__=='__main__':
    main()
'''
def main(s):
    times_L,time_energyL =mfcc.stft(s)
    n,x1,x2 = mfcc.mfcc(times_L,time_energyL,s) #待识别汉字的笔画个数，频域特征提取
    print(n)
    tdoa.tdoa(s,x1,x2) #时域特征提取
    #使用训练好的模型进行笔画识别
    reader_mfcc =modlewrite.data_read_csv('mfcc.csv') #文件中是待识别汉字的特征值
    reader_mfcc_tdoa = modlewrite.data_read_csv('tdoa.csv')

    X_mfcc = modlewrite.feature_read(reader_mfcc,n,num*16*3+16*2) #读取特征数据
    X_tdoa =  modlewrite.feature_read(reader_mfcc_tdoa,n,num)

    X = np.hstack([X_tdoa,X_mfcc]) #时频域特征组合
    y_hat_mfcc= modlewrite.tree_read(X_mfcc)
    y_hat_mfcc_tdoa = modlewrite.tree_mfcc_tdoa_read(X)

    print('仅频域特征识别笔画结果：',y_hat_mfcc)
    print('时频域特征组合识别笔画结果：',y_hat_mfcc_tdoa)

    #笔画识别结果转化为字典集相同形式
#    result_mfcc = category(y_hat_mfcc)
#    result_tdoa = category(y_hat_mfcc_tdoa)

#    print ("仅频域特征识别结果：",result_mfcc)
#    print ("时频域特征组合识别结果：",result_tdoa)

    reader_data =modlewrite.data_read_csv('data1.csv') #数据库文件
    word_mfcc =[[] for i in range (12)]
    word_mfcc_tdoa =[[] for i in range (12)]

#对比数据库结果
    for x in reader_data:
        y = x[0]
        x.remove(x[0])
        word_mfcc = contrast(y_hat_mfcc,x,y,word_mfcc)
        word_mfcc_tdoa= contrast(y_hat_mfcc_tdoa,x,y,word_mfcc_tdoa)

    word_mfcc = arrchange(word_mfcc)
    word_mfcc_tdoa = arrchange(word_mfcc_tdoa)


    if len(word_mfcc)==0:
        print("mfcc分类失败！")
    if len(word_mfcc_tdoa)==0 and len(word_mfcc) == 0:
        print("mfcc_tdoa分类失败！")
        return
    else:
        if 0 <len(word_mfcc) <5:
            print("仅频域特征识别汉字：")
            print(word_mfcc)
        if  len(word_mfcc) >= 5:
            print(word_mfcc[0],word_mfcc[1],word_mfcc[2],word_mfcc[3],word_mfcc[4])

        if  len(word_mfcc_tdoa) <5:
            print("时频域特征组合识别汉字：")
            print(word_mfcc_tdoa)
        if len(word_mfcc_tdoa) >=5:
            print(word_mfcc_tdoa[0], word_mfcc_tdoa[1], word_mfcc_tdoa[2], word_mfcc_tdoa[3], word_mfcc_tdoa[4])

if __name__=='__main__':
    s = 'pen2word/9.wav'
    main(s)