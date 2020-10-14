import itertools
import os
import threading
import time
from math import *

import cv2
import GPUtil as GPU
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import psutil
import skimage.io as io
import torch
import tqdm
from skimage import color

from models.models import create_model
from options.train_options import TrainOptions
from util import html
from util.metrics import *
from util.visualizer import Visualizer

opt = TrainOptions().parse()

model = create_model(opt)

#TEST 
#Only for one image dataset
data_dir = opt.dataroot + "/test/0/"
print(data_dir)
data_n = len(os.listdir(data_dir))/2

sum_iou = sum_dice = sum_acu = sum_prec = sum_rec = 0
sum_iouK = sum_diceK = sum_acuK = sum_precK = sum_recK = 0

##THREAD FOR GPU USAGE 
GPUs = GPU.getGPUs()
gpu = GPUs[0]

def worker(arg):
    t = threading.currentThread()
    while getattr(t,"do_run",True):
        listCPU = getattr(t,"CPUListUsage")
        listGPU = getattr(t,"GPUListUsage")
        process = psutil.Process(os.getpid())
        usageCPU = process.memory_percent()/100
        usageGPU = gpu.memoryUtil
        listCPU.append(usageCPU)
        listGPU.append(usageGPU)
        #print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available), "Proc size: " + humanize.naturalsize( process.memory_info().rss))
        #print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

##Lists that contain ALL values
results = {
    'ext_kmean': [],
    'ext_mscnn': [],
    
    'gpu_usage_mscnn': [],
    'cpu_usage_mscnn': [],
    
    'gpu_usage_kmean': [],
    'cpu_usage_kmean': [],
    
    'iou_ms_class':[],
    'iou_km_class':[]
}

for i in tqdm.tqdm(range(int(data_n))):
    img = io.imread(data_dir + str(i) + '.jpg')
    img_K = cv2.imread(data_dir + str(i) + '.jpg') #FOR K-MEANS, DOESN'T WORK WITHO NORMAL IO.IMREAD
    img_RGB = cv2.cvtColor(img_K,cv2.COLOR_BGR2RGB)
    
    img_L = cv2.imread(data_dir + str(i) + '.tiff')
    img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
    #img = img/255
    img_RGB = img_RGB/255
    
    data = np.expand_dims(img_RGB, axis=0)
    data = (np.transpose(data, (0, 3, 1, 2)))
    data = torch.from_numpy(data.copy())
    data = data.to(device='cuda',dtype=torch.float)
    
    ##START THREAD
    t = threading.Thread(target=worker, args=("task",))
    t.GPUListUsage = []
    t.CPUListUsage = []
    t.start() 
    start_time = time.time() #EXECUTION TIME
    with torch.no_grad():
        prediction = model.net(data)
    t.do_run=False
    t.join() #end thread
    exect_time_ms = time.time() - start_time
    # add to execution time list
    results['ext_mscnn'].append(exect_time_ms)
    results['gpu_usage_mscnn'].append(sum(t.GPUListUsage)/len(t.GPUListUsage))
    results['cpu_usage_mscnn'].append(sum(t.CPUListUsage)/len(t.CPUListUsage))
    
    out = torch.argmax(prediction, 1) #Index array of prediction
    out = out.cpu().numpy()
    out = np.squeeze(out)
    
    values = np.unique(out)
    combinations = np.array(list(itertools.permutations(range(0,len(values)))))
    
    ground_truth = indexArray(img_L) #convert tiff to array of indices
    metrics = getMetrics(out+3,ground_truth,values+3,combinations)
    
    #We get K value for K-means for ground-truth
    k = len(np.unique(ground_truth))
    
    ##START THREAD
    t = threading.Thread(target=worker, args=("task",))
    t.GPUListUsage = []
    t.CPUListUsage = []
    t.start()
    start_time = time.time() #EXECUTION TIME
    kmeans_result = K_means(img_K,k)
    t.do_run=False
    t.join() #end thread
    exect_time_ms = time.time() - start_time
    # Add to execution time list
    results['ext_kmean'].append(exect_time_ms)
    results['gpu_usage_kmean'].append(sum(t.GPUListUsage)/len(t.GPUListUsage))
    results['cpu_usage_kmean'].append(sum(t.CPUListUsage)/len(t.CPUListUsage))
    
    values_k = np.unique(kmeans_result).astype(int)
    combinations_k = np.array(list(itertools.permutations(range(0,len(values_k)))))
    k_metrics = getMetrics(kmeans_result+3,ground_truth,values_k+3,combinations_k) # add three to avoid overlapping combinations
    
    sum_iou+=metrics["iou"]
    sum_dice+=metrics["dice"]
    sum_acu+=metrics["acu"]
    sum_prec+=metrics["prec"]
    sum_rec+=metrics["rec"]
    
    results["iou_ms_class"].append(metrics["iou_class"])
    results["iou_km_class"].append(k_metrics["iou_class"])
    
    sum_iouK+=k_metrics["iou"]
    sum_diceK+=k_metrics["dice"]
    sum_acuK+=k_metrics["acu"]
    sum_precK+=k_metrics["prec"]
    sum_recK+=k_metrics["rec"]
    
    #IF WE WANT TO PRINT THE PREDICTION LABEL
    #label = color.label2rgb(out)
    #plt.imshow(label)
    #plt.show()
ms_pc_values = np.array(results["iou_ms_class"])
km_values = np.array(results["iou_km_class"])
print("MSCnn per class:",np.mean(ms_pc_values,axis=0))    
print("MSCnn per class:",np.mean(km_values,axis=0))    

print('MSCnn iou: ',sum_iou/data_n)
print('MSCnn dice: ',sum_dice/data_n)
print('MSCnn acu: ',sum_acu/data_n)
print('MSCnn prec: ',sum_prec/data_n)
print('MSCnn rec: ',sum_rec/data_n)

print('K-mean iou: ',sum_iouK/data_n)
print('K-mean dice: ',sum_diceK/data_n)
print('K-mean acu: ',sum_acuK/data_n)
print('K-mean prec: ',sum_precK/data_n)
print('K-mean rec: ',sum_recK/data_n)

print('Mumford Shah CNN Execution Time',sum(results['ext_mscnn'])/len(results['ext_mscnn']))
print('KMean Execution Time:', sum(results['ext_kmean'])/len(results['ext_kmean']))

print('CPU Usage MSCnn: ',sum(results['cpu_usage_mscnn'])/len(results['cpu_usage_mscnn']))
print('GPU Usage MSCnn: ', sum(results['gpu_usage_mscnn'])/len(results['gpu_usage_mscnn']))

print('CPU Usage KMean',sum(results['cpu_usage_kmean'])/len(results['cpu_usage_kmean']))
print('GPU Usage KMean', sum(results['gpu_usage_kmean'])/len(results['gpu_usage_kmean']))
