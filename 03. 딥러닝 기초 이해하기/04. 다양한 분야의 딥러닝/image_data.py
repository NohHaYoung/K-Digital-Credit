import  pandas as pd
import numpy as np

import PIL
import matplotlib.image as img
import matplotlib.pyplot as plt

from elice_utils import EliceUtils
elice_utils = EliceUtils()

# 이미지 목록을 불러오는 함수입니다.

def load_data(path):
    return pd.read_csv(path)

'''
1. PIL.Image를 이용하여 
   이름(경로+이름)을 바탕으로 이미지를 불러오고,
   이를 리스트 'images'에 추가하는 함수를 완성합니다.
   main 함수에서 'path'와 'names' 변수를 확인해보세요.
'''

def load_images(path, names):
    
    images=[]
    
    for name in names:
        images.append(PIL.Image.open(path + name))
    
    return images

'''
2. 이미지의 사이즈를 main 함수에 있는 'IMG_SIZE'로 
   조정하고, 이를 Numpy 배열로 변환하는 함수를 완성합니다.
'''

def images2numpy(images, size):
    output = []
    
    for image in images:
        output.append(np.array(image.resize(size)))
    
    return output


# 이미지에 대한 정보를 나타내주는 함수입니다.

def sampleVisualize(np_images):

    fileName = "./data/images/1000092795.jpg"
    ndarray = img.imread(fileName)
    
    plt.imshow(ndarray)
    plt.show()    
    plt.savefig("plot.png")
    
    print("\n1-1. 'fileName' 이미지(원본): ")
    elice_utils.send_image("plot.png")
    
    print('\n1-2. Numpy array로 변환된 원본 이미지:', ndarray)
    print('\n1-3. Numpy array로 변환된 원본 이미지의 크기:', np.array(ndarray).shape)
    
    plt.imshow(np_images[0])
    plt.show()
    plt.savefig("plot_re.png")
    
    print("\n2-1. 'fileName' 이미지(resize 후): ")
    elice_utils.send_image("plot_re.png")
    
    print('\n2-2. Numpy array로 변환된 resize 후 이미지:', np_images[0])
    print('\n2-3. Numpy array로 변환된 resize 후 이미지 크기:', np.array(np_images[0]).shape)    
    
    print('\n3. Numpy array로 변환된 resize 후 이미지 10장의 크기:', np.array(np_images).shape)

'''
3. main 함수를 완성하세요.

   Step01. 이미지를 불러오는 함수를 이용해 
           'images'를 정의합니다.
   
   Step02. 이미지를 Numpy 배열로 바꾸는 함수를 이용해
           'np_images'를 정의합니다.
'''

def main():
    
    CSV_PATH = "./data/data.csv"
    IMG_PATH = "./data/images/"
    IMG_SIZE = (300,300)
    MAX_LEN = 30
    BATCH_SIZE = 2
    
    name_caption = load_data(CSV_PATH)
    names = name_caption['file_name']
    
    images = load_images(IMG_PATH, names)
    np_images = images2numpy(images, IMG_SIZE)
    
    sampleVisualize(np_images)
    
    return images, np_images
    
if __name__=='__main__':
    main()