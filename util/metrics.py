from PIL import Image  
import PIL  
from sklearn.metrics import accuracy_score,jaccard_score,precision_score,recall_score
import numpy as np
import cv2
from skimage.color import rgb2gray,label2rgb

def K_means(image,k):
    
    img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    attempts=100
    ret,label,center=cv2.kmeans(vectorized,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    
    res2gray = rgb2gray(result_image)
    values = np.unique(res2gray)
    #Create index array for label
    index = 0
    for e in values:
        res2gray[res2gray==e] = index
        index+=1

    return res2gray

def indexArray(image):
    #Values from labels to gray
    #Unique values
    #THE SIZE OF THIS LIST SHOULD BE THE SAME SIZE OF THE UNIQUE RECOGNIZED COLORS IN THE IMAGE 
    lab_colors = {
        "blue":[94/255,0,1],
        "orange":[1,106/255,0],
        "yellow":[1,238/255,0],
    }
    
    g_conv = [0.2125,0.7154,0.0721] #Values from skimage
    unique_value = []
    unique_value.append(lab_colors["blue"] @ np.array(g_conv))
    unique_value.append(lab_colors["orange"] @ np.array(g_conv))
    unique_value.append(lab_colors["yellow"] @ np.array(g_conv))
    unique_value = np.array(unique_value)
    
    image = image/255
    gray_img = rgb2gray(image)

    count = 0
    for v in unique_value:
        gray_img[gray_img==v]=count
        count+=1
    
    unique_colors, count_colors = np.unique(gray_img,return_counts=True)
    perc = []
    freq = np.sum(count_colors)
    for i in count_colors:
        perc.append((i/freq)*100)
    assert sum(perc) == 100
    return gray_img

def getMetrics(prediction,target,unique_values,combinations):
    max_iou = max_dice = max_acu = max_prec = max_rec = 0
  
    #For each combinations we test the scores
    for c in combinations:
        auxPred = prediction.copy()
        index = 0
        for v in unique_values:
            auxPred[auxPred==v] = c[index]
            index+=1

        iou =  jaccard_score(auxPred.flatten(), target.flatten(),average=None)
        dice = np.divide(2*iou,1+iou)

        iouf = jaccard_score(auxPred.flatten(), target.flatten(), average='macro')
        dicef = dice.mean()
        acuf= accuracy_score(auxPred.flatten(), target.flatten())
        precf = precision_score(auxPred.flatten(), target.flatten(), average='macro')
        recf = recall_score(auxPred.flatten(), target.flatten(), average='macro')
        
        if(iouf + dicef + acuf + precf + recf > max_iou + max_dice + max_acu + max_prec + max_rec): 
            max_iou = iouf
            max_dice = dicef
            max_acu = acuf
            max_prec = precf
            max_rec = recf
    
    metrics = {
        "iou":max_iou,
        "dice":max_dice,
        "acu":max_acu,
        "prec":max_prec,
        "rec":max_rec
    }
    return metrics