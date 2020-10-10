import time
import os
import numpy as np
from options.train_options import TrainOptions
from LiTS_getDatabase_unet import DataProvider_LiTS
from models.models import create_model
from util.visualizer import Visualizer
from math import *
from util import html
import ipdb
#Added imports
import torch
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import color
import cv2
import itertools
from util.metrics import *
import psutil
import os, time
import GPUtil as GPU
import threading
import tqdm

opt = TrainOptions().parse()

input_min = 0
input_max = 400

data_train = DataProvider_LiTS(opt.inputSize, opt.fineSize, opt.segType, opt.semi_rate, opt.input_nc,
                               opt.dataroot, a_min=input_min, a_max=input_max, mode="train")
dataset_size = data_train.n_data
print('#training images = %d' % dataset_size)
training_iters = int(ceil(data_train.n_data/float(opt.batchSize)))

total_steps = 0
model = create_model(opt)
visualizer = Visualizer(opt)
#TRAINING
if opt.isTrain:
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()

        """ Train """
        for step in range(1, training_iters+1):
            batch_x, batch_y, path = data_train(opt.batchSize)
            data = {'A': batch_x, 'B': batch_y,
                    'A_paths': path, 'B_paths': path}
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += 1
            model.set_input(data)
            model.optimize_parameters()
            #model.save('latest') just to load model later

            if step % opt.display_step == 0:
                save_result = step % opt.update_html_freq == 0
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result)

                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(
                    epoch, step, training_iters, errors, t, 'Train')

            if step % opt.plot_step == 0:
                if opt.display_id > 0:
                    visualizer.plot_current_errors(
                        epoch, step / float(training_iters), opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_learning_rate()
        
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
    
}

for i in tqdm.tqdm(range(int(data_n))):
    img = io.imread(data_dir + str(i) + '.jpg')
    img_K = cv2.imread(data_dir + str(i) + '.jpg') #FOR K-MEANS, DOESN'T WORK WITHO NORMAL IO.IMREAD
    img_L = cv2.imread(data_dir + str(i) + '.tiff')
    img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
    img = img/255

    data = np.expand_dims(img, axis=0)
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
    ##Add to execution time list
    results['ext_mscnn'].append(exect_time_ms)
    results['gpu_usage_mscnn'].append(sum(t.GPUListUsage)/len(t.GPUListUsage))
    results['cpu_usage_mscnn'].append(sum(t.CPUListUsage)/len(t.CPUListUsage))
    
    out = torch.argmax(prediction, 1) #Index array of prediction
    out = out.cpu().numpy()
    out = np.squeeze(out)
    
    values = np.unique(out)
    combinations = np.array(list(itertools.permutations(range(0,len(values)))))
    
    ground_truth = indexArray(img_L) #convert tiff to array of indices
    metrics = getMetrics(out,ground_truth,values,combinations)
    
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
    ##Add to execution time list
    results['ext_kmean'].append(exect_time_ms)
    results['gpu_usage_kmean'].append(sum(t.GPUListUsage)/len(t.GPUListUsage))
    results['cpu_usage_kmean'].append(sum(t.CPUListUsage)/len(t.CPUListUsage))
    
    values_k = np.unique(kmeans_result).astype(int)
    combinations_k = np.array(list(itertools.permutations(range(0,len(values_k)))))
    k_metrics = getMetrics(kmeans_result,ground_truth,values_k,combinations_k)
    
    sum_iou+=metrics["iou"]
    sum_dice+=metrics["dice"]
    sum_acu+=metrics["acu"]
    sum_prec+=metrics["prec"]
    sum_rec+=metrics["rec"]
    
    sum_iouK+=k_metrics["iou"]
    sum_diceK+=k_metrics["dice"]
    sum_acuK+=k_metrics["acu"]
    sum_precK+=k_metrics["prec"]
    sum_recK+=k_metrics["rec"]
    
    #IF WE WANT TO PRINT THE PREDICTION LABEL
    #label = color.label2rgb(out)
    #plt.imshow(label)
    #plt.show()

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
