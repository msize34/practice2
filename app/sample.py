import numpy as np
from PIL import Image
import cv2
import pickle
import glob as gb
from matplotlib import pyplot as plt

def preProcess(file):
    img=Image.open(file)
    img=img.convert('L')
    img=np.asarray(img)
    img=cv2.resize(img,dsize=(32,32))
#     ret,img = cv2.threshold(img,130,255,cv2.THRESH_BINARY)

    img=img.reshape(-1)
    img=img/255
    return img

data=gb.glob('E:\\ユーザーデータ\\ピクチャ\\*3.jpg')
img = preProcess(data[0])
load_model=pickle.load(open('../skmodel.sav', 'rb'))
img=img.reshape(1,1024)
print(img.shape)
result=load_model.predict(img)
print(result)
pic=img[0]
pic=pic.reshape(32,32)
print(pic.shape)
plt.imshow(pic)
plt.show()

pwd