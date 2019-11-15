"""
Reference: https://towardsdatascience.com/yolov2-object-detection-using-darkflow-83db6aa5cf5f
Author: Lucas S. Althoff
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import os, os.path
import glob

#Boxing function
def box(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
            
    return newImage

#Detector options 
#Tiny YOLO ativar rede
options = {"model": "cfg/v1.1/tiny-yolov1.cfg", 
           "load": "bin/tiny-yolov1.weights", 
           "threshold": 0.1, 
          }

#YOLO – Video options
options = {"model": "cfg/yolo.cfg", 
           "load": "C:/Users/usuario/Desktop/YOLO WEIGHTS/yolo.weights", 
           "threshold": 0.1,
           "demo": "mariobros_360p.mp4","saveVideo" == True}

#Tiny YOLO - Video
#python ./flow --model cfg/v1.1/tiny-yolov1.cfg --load bin/tiny-yolov1.weights --demo baloes_pessoas_carros.mp4 --saveVideo --threshold 0.1
options = {"model": "cfg/v1.1/tiny-yolov1.cfg", 
           "load": "bin/tiny-yolov1.weights", 
           "threshold": 0.1,
           "demo": "cavalos_humanos.mp4",
           "saveVideo" == True}

tfnet = TFNet(options)

#Teste em imagem
path = r'C:/Users/usuario/Google Drive/Acadêmico/Doutorado/2019-1/ProcessamentoImagens/Trabalho2-Especificação de Histograma por Dataset/dataset4/'
name_img = r'i146.jpg'

img = plt.imread(path + name_img)
results = tfnet.return_predict(img)
img_out = boxing(img, results)
_, ax = plt.subplots(figsize=(20, 10))
ax.imshow(img_out)
plt.show()

#Aplicando YOLO em múltiplas imagens
path = r'C:/Users/usuario/Google Drive/Acadêmico/Doutorado/2019-1/ProcessamentoImagens/Trabalho2-Especificação de Histograma por Dataset/dataset4/'
path_out = 'C:/Users/usuario/Desktop/YOLO_WEIGHTS/resultados'
num_img = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]) 
filenames = glob.glob(path+r'//*')

for nome_img in filenames:
    print("Detectando objetos na Imagem:" + nome_img)
    img = plt.imread(nome_img)
    results = tfnet.return_predict(img)
    img_out = box(img, results)
    _, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(img_out)
    plt.imsave(path_out +"/thres01_tiny_" + nome_img.split('t4\\')[1], img_out)
