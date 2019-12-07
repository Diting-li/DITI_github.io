# -*-coding:utf-8-*-
"""
@Author: Chunyang Lee
@email: chuny6896@163.com
@software: garner
@file: Extract_prammaters.py
@CreatDate: 2019/11/19 11:08
@Description: 
"""
import re

def Extract_ptammaters(pramaters_name):
    file_handle=open(pramaters_name+'.txt',mode='r')
    prammaters_mumber_count = file_handle.read(10)
    prammaters_mum = re.findall(r"\d+\.?\d*",prammaters_mumber_count)
    prammaters_mum = int(prammaters_mum[0])
    type = file_handle.read(prammaters_mum*10)
    list_sample_type = re.findall(r"\d+\.?\d*",type)
    lis = []
    for i in range(prammaters_mum):
        number = int(list_sample_type[i])
        lis.append(number)
    return lis