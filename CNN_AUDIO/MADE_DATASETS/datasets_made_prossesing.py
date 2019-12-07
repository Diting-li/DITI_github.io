'''本代码是为了检验CNN四层网络而制作的1024大小样本的训练集'''

import struct
import os
import random
import gzip
import re
from scipy.fftpack import fft
import pandas as pd
class Madedatasets:
    def __init__(self):
        self.train = "train"
        self.test = "test"
    @staticmethod
    def Extract_wav_number(Data_Binary_dir_txtname):
        file_handle = open("MADE_DATASETS/Data_Binary" + "/" + Data_Binary_dir_txtname + ".txt", mode='r')
        content = int(file_handle.read(20))
        return content
    def get_pramaters(self,train_or_test):
        train_normal_number = self.Extract_wav_number("Binary_" + train_or_test + "_normal") #(160)
        train_abnormal_number = self.Extract_wav_number("Binary_"+ train_or_test +"_abnormal") #(353)
        label_data_number = data_all_data_mnumber = train_normal_number + train_abnormal_number
        file_handle=open('sample_type.txt',mode='r')
        type = file_handle.read(20)
        list_sample_type = re.findall(r"\d+\.?\d*",type)
        data_high_number = int(list_sample_type[0])
        data_wide_number = int(list_sample_type[1])
        return train_normal_number,train_abnormal_number,label_data_number,data_all_data_mnumber,data_high_number,data_wide_number


    "Datasets_Finished_Finally_output/Input_Data"
    '公司：训练二进制/Normal_train_highway'

    @staticmethod
    def Pir_made_dataset(self,str_train_or_test):
        train_normal_number,train_abnormal_number,label_data_number,data_all_data_mnumber,data_high_number,data_wide_number = self.get_pramaters(str_train_or_test)
        data_one_aix = data_high_number * data_wide_number
        # 编写导出的数据集文件夹路径。什么，没有？创建一个then！
        pathdir = 'Datasets_Finished_Finally_output/Input_Data'
        pt = os.path.exists(pathdir + "/")
        if pt:
            print("File Exist!")
        else:
            os.mkdir(pathdir + "/")
        # 以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
        data_magic_number = 3331
        label_magic_number = 2049

        fid0 = open(pathdir + "/" + str_train_or_test +"-audio-idx3-ubyte", "wb")
        fid1 = open(pathdir + "/" + str_train_or_test +"-labels-idx1-ubyte", "wb")
        # 写入文件头
        MAG = struct.pack('>iiii', data_magic_number, data_all_data_mnumber, data_high_number, data_wide_number)
        MBG = struct.pack('>ii', label_magic_number, label_data_number)
        fid0.write(MAG)
        fid1.write(MBG)
        # 建立空list，存放数据和标签数据
        lis = []
        lis_lable = []
        #####################################################################
        # 写入正常数据（数据和标签），即：写入类别1的数据
        dir = "MADE_DATASETS/Data_Binary/Binary_"+ str_train_or_test + "_normal"
        index = 0
        binfile = open(dir, 'rb')  # 以二进制方式打开文件
        buf = binfile.read()  # 把数据都读进来
        binfile.close()

        for e in range(train_normal_number):
            images = []#先设一个空的list将下面提取的样本数据放到里面，由于上面还有个for语句，每次结束之后会清空这个list

            for h in range(data_one_aix):#在这个for语句之内要做的一件事是：把一个声音样本数据写进去
                img, = struct.unpack_from('>f', buf, index)
                index += struct.calcsize('4B')
                images.append(img)
                index = index

            lis.append(images)#把这个样本数据放到大的list里面，将来要混合、打乱顺序
        for i in range(train_normal_number):
            lab_num = 1
            lis_lable.append(lab_num)
        ###################################################################
        # 写入异常数据（数据和标签），即：写入类别2的数据
        dir_ab = "MADE_DATASETS/Data_Binary/Binary_"+ str_train_or_test + "_abnormal"

        index = 0
        binfile = open(dir_ab, 'rb')  # 以二进制方式打开文件
        buf = binfile.read()  # 把数据都读进来
        binfile.close()
        # if A1 > 0:
        #     index = (A1 * 1250) * 4
        # else:
        #     index = 0
        for r in range(train_abnormal_number):
            images = []
            for t in range(data_one_aix):
                img, = struct.unpack_from('>f', buf, index)
                index += struct.calcsize('4B')
                images.append(img)
                index = index
            lis.append(images)
        for g in range(train_abnormal_number):
            lab_num = 0
            lis_lable.append(lab_num)

        ###########################################################
        # 混合两类数据并且打乱顺序
        DATA_comb = list(zip(lis, lis_lable))
        random.Random().shuffle(DATA_comb)
        lis_shuffle, lis_lable_shuffle = zip(*DATA_comb)
        lis_shuffle_done = []
        for x in lis_shuffle:
            lis_shuffle_done += x
        num_bits = data_all_data_mnumber * data_one_aix
        for n in range(num_bits):
            data_bin_images = struct.pack('>f', lis_shuffle_done[n])
            # print(data_bin_images)
            fid0.write(data_bin_images)
        fid0.close()
        for l in range(data_all_data_mnumber):
            data_bin_labels = struct.pack('>b', lis_lable_shuffle[l])
            fid1.write(data_bin_labels)
        fid1.close()
        print("the %d st train_data and labels files have been finished")
        ######################## 制作数据集格式 ##################################
        f_in = open(pathdir + "/" +  str_train_or_test +"-audio-idx3-ubyte", 'rb')
        f_out = gzip.open(pathdir + "/" + str_train_or_test +"-audio-idx3-ubyte" + ".gz", 'wb')
        f_out.writelines(f_in)
        f_out.close()
        f_in.close()
        os.remove(pathdir + "/"  + str_train_or_test +"-audio-idx3-ubyte")
        f_in = open(pathdir + "/" + str_train_or_test +"-labels-idx1-ubyte", 'rb')
        f_out = gzip.open(pathdir + "/" + str_train_or_test +"-labels-idx1-ubyte" + ".gz", 'wb')
        f_out.writelines(f_in)
        f_out.close()
        f_in.close()
        os.remove(pathdir + "/" + str_train_or_test +"-labels-idx1-ubyte")

    def Made_datasets(self):
        self.Pir_made_dataset(self,self.train)
        self.Pir_made_dataset(self,self.test)

