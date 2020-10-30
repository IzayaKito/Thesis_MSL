from __future__ import print_function

import argparse
import collections
import inspect
import os
import re
import io
import torchvision.transforms as transforms

import numpy as np
import torch
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
import pickle
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array


def tensor2im(image_tensor, imtype=np.float32):
    image_numpy = image_tensor[0].cpu().float().numpy()
    imgSize = image_numpy.shape

    if len(imgSize) == 2:  # ONLY WANTS TO PRINT ONE CHANNEL OF IMAGE?? (THE TENSOR SENT WAS ONE CHANNEL)
        image_numpy = image_numpy.reshape(1, imgSize[0], imgSize[1])
    # elif len(imgSize) == 3:
    #    image_numpy = image_numpy[1].reshape(1, imgSize[1], imgSize[2])

    #image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    # image_pil = Image.fromarray(image_numpy)
    # image_pil = Image.fromarray(image_numpy.astype(np.uint8))
    # image_pil.save(image_path)
    # print('save_image')
    pass


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(
        getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(512),
                                        transforms.CenterCrop(512),
                                        transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes, model):
    tensor = transform_image(image_bytes)
    tensor = tensor.to(device='cuda', dtype=torch.float)
    print(tensor.size)

    with torch.no_grad():
        prediction = model.net(tensor)

    out = torch.argmax(prediction, 1)  # Index array of prediction
    out = out.cpu().numpy()
    out = np.squeeze(out)

    image = color.label2rgb(
        out, colors=[[1, 238/255, 0], [1, 106/255, 0], [94/255, 0, 1]])
    return image


def saveImageStatic(app_root, filename, image):
    plt.imsave(os.path.join(app_root, "static/" +
                            filename), image)


def deleteImageStatic(app_root, filename):
    os.remove(os.path.join(app_root, "static/" +
                           filename))

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def read_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj