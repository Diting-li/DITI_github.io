"""本脚本是用来从声音（.wav）中提取mfcc特征并将数据（（299*13*count），count是wav文件数），并存入二进制文件中，说明：数字是double类型不是float；"""

import scipy.io.wavfile as wav
import matplotlib.pylab as plt
from python_speech_features import mfcc
import operator
from functools import reduce
import os
import struct

Wav_input_dir = ['MADE_DATASETS/Data_wav_trainsets_Normal/', 'MADE_DATASETS/Data_wav_trainsets_Abnormal/',
                 'MADE_DATASETS/Data_wav_testsets_Normal/', 'MADE_DATASETS/Data_wav_testsets_Abnormal/']
list_root_dir = Wav_input_dir
Binary_output_dir = "MADE_DATASETS/Data_Binary"
class Wav_to_Binary:
    def __init__(self):
        self.Binary_output_dir = Binary_output_dir
        self.list_root_dir = list_root_dir
        self.Wav_input_dir = Wav_input_dir
        self.Binary_output_dir = Binary_output_dir
    @staticmethod
    def Binary(Wav_input_dir,Binary_name,Binary_output_dir):
        # 检查/创建 将要生成的二进制文件上级目录（在哪个文件夹下）
        pathdir_bigdata = Binary_output_dir #脚本目录下 "异常_训练集“ 文件夹
        b = os.path.exists(pathdir_bigdata)
        if b:
            print("File Exist!")
        else:
            os.mkdir(pathdir_bigdata)
        filename = Binary_name #二进制文件名
        path = Wav_input_dir    #获取当前路径（声音文件.wav 输入）
        count = 0 #为计算该目录下有多少个wav文件计数
        for root,dirs,files in os.walk(path):    #遍历统计wav文件数
              for each in files:
                     count += 1   #统计文件夹下文件个数
        #print(count)
        m = count
        list_all = [] #每个299*13=3887 将所有的mfcc数据存储在list_all里
        #循环每个声音文件，提取mfcc数据
        for i in range(m):
            fs, audio = wav.read(path + str(i+1) + ".wav")
            feature_mfcc = mfcc(audio, samplerate=fs)
            mfcc_features = feature_mfcc.T
            mfcc_features = mfcc_features.tolist()
            y = mfcc_features
            y_ = reduce(operator.add, y)
            #plot梅尔倒谱系数图
            # plt.matshow(mfcc_features)
            #     # plt.title('MFCC')
            #     # plt.show()
            list_all.append(y_)#[:3887]
        #由于是[[1，2，3]，[4,5,6]...[7,8,9]]类似的格式，需转换成[1,2,3....7,8,9]格式，用reduce函数转换
        y_all = reduce(operator.add, list_all)
        ##print(y_all[:6])
        LONG = len(y_all)
        print(LONG)
        #循环list里的每一个元素写入
        fid = open(pathdir_bigdata + "/" + filename, 'wb') #创建写入文件（二进制文件）
        for n in range(LONG):
            data_bin_images = struct.pack('>d', y_all[n])
            fid.write(data_bin_images)
        fid.close()
        return m

    @staticmethod
    def pramaters(dir,wav_nameber_pramater):
        file_handle = open(dir, mode='w')
        file_handle.write(str(wav_nameber_pramater))
        file_handle.close()

    def Main(self):
        for lis in list_root_dir:
        #print(list_root_dir[:])
            if (list_root_dir.index(lis) + 1) == 1:
                filename = "Binary_train_normal"
                #self.Binary(lis,filename,Binary_output_dir)
                self.pramaters(Binary_output_dir+'/'+filename+'.txt',self.Binary(lis,filename,Binary_output_dir))

            else:
                pass
            if (list_root_dir.index(lis) + 1) == 2:
                filename = "Binary_train_abnormal"
                #self.Binary(lis,filename,Binary_output_dir)
                self.pramaters(Binary_output_dir + '/' + filename + '.txt',
                               self.Binary(lis, filename, Binary_output_dir))
            else:
                pass
            if (list_root_dir.index(lis) + 1) == 3:
                filename = "Binary_test_normal"
                #self.Binary(lis,filename,Binary_output_dir)
                self.pramaters(Binary_output_dir + '/' + filename + '.txt',
                               self.Binary(lis, filename, Binary_output_dir))
            else:
                pass
            if (list_root_dir.index(lis) + 1) == 4:
                filename = "Binary_test_abnormal"
                #self.Binary(lis,filename,Binary_output_dir)
                self.pramaters(Binary_output_dir + '/' + filename + '.txt',
                               self.Binary(lis, filename, Binary_output_dir))
            else:
                pass


