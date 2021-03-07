import os
import struct
import numpy as np
import math
from PIL import Image
learn_rate=0.03#学习率


def get_data_lable():

    with open(r'C:\Users\Administrator\Downloads\train-labels.idx1-ubyte', 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8)) 

        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(r'C:\Users\Administrator\Downloads\train-images.idx3-ubyte', 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))

        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
 

    list_labels=np.zeros([60000,10])
    for i3 in range(60000):
        list_labels[i3][labels[i3]]=1

    return list_labels,images
        
def parameter_initialization():
    w1 = 0.1*np.random.rand(784,40) 
    w1 -= np.mean(w1)
    w2 = 0.1*np.random.rand(40,60) 
    w2 -= np.mean(w2)
    w3 = 0.1*np.random.rand(60,10) 
    w3 -= np.mean(w3)

    b1 = np.random.randn(1,40)
    b1 -= np.mean(b1)
    b2 = np.random.randn(1,60)
    b2 -= np.mean(b2)
    b3 = np.random.randn(1,10)
    b3 -= np.mean(b3)

    return w1,w2,w3,b1,b2,b3


def softmax(z):
	return 1 / (1 + np.exp(-z))
 

def buildmode(images,list_labels,w1,w2,w3,b1,b2,b3):
     #正向传播
    for i2 in range(60000): 
        GET = images[i2].reshape((1,784))
        label = list_labels[i2].reshape((1,10))
        cal_one = softmax(np.add(np.dot(GET,w1),b1))#输入层输出
        cal_two = softmax(np.add(np.dot(cal_one,w2),b2))#隐二层输出w
        output  = softmax(np.add(np.dot(cal_two,w3),b3))

        out_sum=1/2*((np.subtract(label,output))**2)


        a = np.multiply(output, np.subtract(1,output))

        g = np.multiply(a,output-label)

        b = np.dot(g, np.transpose(w3))

        c = np.multiply(cal_two, np.subtract(1,cal_two))

        e = np.multiply(b,c)

        d = np.dot(e,np.transpose(w2))
        
        f = np.multiply(cal_one, np.subtract(1,cal_one))

        h = np.multiply(d,f)

        w3 = w3 - learn_rate*np.dot(np.transpose(cal_two),g) 
        w2 = w2 - learn_rate*np.dot(np.transpose(cal_one),e)
        w1 = w1 - learn_rate*np.dot(np.transpose(GET),h)

        b3 = b3 - learn_rate*g
        b2 = b2 - learn_rate*e
        b1 = b1 - learn_rate*h
        if i2 % 50==0:
            print('第{a}张训练的当前的loss值为{b}'.format(a=i2,b=np.sum(out_sum)))
    return w1,w2,w3,b1,b2,b3

def save_w_and_b(w1,w2,w3,b1,b2,b3):
    np.savez('update_w_and_b',W1=w1,W2=w2,W3=w3,B1=b1,B2=b2,B3=b3)
    

def get_test_data():
    
    with open(r'C:\Users\Administrator\Downloads\t10k-labels.idx1-ubyte', 'rb') as lbpath2:
        magic, n = struct.unpack('>II',lbpath2.read(8)) 

        labels_test = np.fromfile(lbpath2,dtype=np.uint8)

    with open(r'C:\Users\Administrator\Downloads\t10k-images.idx3-ubyte', 'rb') as imgpath2:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath2.read(16))

        images_test = np.fromfile(imgpath2,dtype=np.uint8).reshape(len(labels_test), 784)

    list_labels_test=np.zeros([10000,10])
    for i4 in range(10000):
        list_labels_test[i4][labels_test[i4]]=1

    return list_labels_test,images_test


def test(labels_test,images_test,w1,w2,w3,b1,b2,b3):
    yes = 0
    for i in range(len(images_test)):
        GET_image = images_test[i].reshape((1,784))
        label = labels_test[i].reshape((1,10))
        cal_one = softmax(np.add(np.dot(GET_image,w1),b1))#输入层输出
        cal_two = softmax(np.add(np.dot(cal_one,w2),b2))#隐二层输出w
        output  = softmax(np.add(np.dot(cal_two,w3),b3))
        if np.argmax(output)== np.argmax(label):
            yes = yes + 1

    return yes/len(labels_test)

    

if __name__ == "__main__":
    #w1,w2,w3,b1,b2,b3=parameter_initialization()
    #labels,images=get_data_lable()
   # for i1 in range(1000):
    #   w1,w2,w3,b1,b2,b3=buildmode(images,labels,w1,w2,w3,b1,b2,b3)
    labels_test,images_test = get_test_data()
   # save_w_and_b(w1,w2,w3,b1,b2,b3)
    data = np.load(r"D:\python code\update_w_and_b.npz")

    W1 = data['W1']
    W2 = data['W2']
    W3 = data['W3']
    B1 = data['B1']
    B2 = data['B2']
    B3 = data['B3']
    last=test(labels_test,images_test,W1,W2,W3,B1,B2,B3)
    print('测试正确率为{c}'.format(c=last))
    




    