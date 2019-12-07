# -*-coding:utf-8-*-
"""
@Author: Chunyang Lee
@email: chuny6896@163.com
@software: garner
@file: RUN.py
@CreatDate: 2019/11/14 17:46
@Description: 
"""
from CNN_trainning.CNN_model import CNN_model
from MADE_DATASETS.datasets_made_prossesing import Madedatasets
from MADE_DATASETS.wav_to_Biary import Wav_to_Binary
if __name__ == '__main__':
    #Wav_to_Binary().Main()
    #Madedatasets().Made_datasets()
    CNN_model().train()
