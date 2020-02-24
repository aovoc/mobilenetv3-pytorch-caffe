import numpy as np
import cv2
import sys
import os
import random
import code
import math
import string
import caffe
import torch
from pathos.multiprocessing import ProcessingPool as Pool
from PIL import Image, ImageDraw
from mobilenetv3 import mobilenetv3_large
from mobilenetv3 import mobilenetv3_small
from mobilenetv3_old import mobilenetv3_large_old
from mobilenetv3_old import mobilenetv3_small_old

np.set_printoptions(threshold=np.inf)

caffe.set_device(0)
caffe.set_mode_gpu()

def read_img(path, im_size, crop_pct=1.0, keep_ratio=True):
    assert os.path.exists(path)
    data = Image.open(path).convert('RGB')
    scale_size = int(math.floor(im_size/crop_pct))
    min_edge = min(data.size[0], data.size[1])
    resize_ratio = scale_size / float(min_edge)
    if keep_ratio:
        data = data.resize((int(round(resize_ratio * data.size[0])), int(round(resize_ratio * data.size[1]))), Image.BICUBIC)
        data = data.crop((int(round((data.size[0] -im_size) / 2.)), int(round((data.size[1] -im_size) / 2.)), \
            int(round((data.size[0] -im_size) / 2.)) + im_size, int(round((data.size[1] -im_size) / 2.)) + im_size ))
    else:
        data.resize((scale_size, scale_size), Image.BICUBIC)
        data = data.crop( (int(round((scale_size -im_size) / 2.)), int(round((scale_size -im_size) / 2.)), \
             int(round((scale_size -im_size ) / 2)) + im_size, int(round((scale_size -im_size) / 2)) + im_size ))
    data = np.asarray(data) / 255.
    data = (np.asarray(data) - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    data = np.transpose(data, (2, 0, 1))
    return data


top5_dict = {'0':0, '1':0}
top1_dict = {'0':0, '1':0}
ptop5_dict = {'0':0, '1':0}
ptop1_dict = {'0':0, '1':0}

if __name__=="__main__":
    src_path = "ILSVRC2012_val.txt"
    
    if sys.argv[1] == "mobilenetv3_small":
        model = mobilenetv3_small(pretrained=True)
    elif sys.argv[1] == "mobilenetv3_large":
        model = mobilenetv3_large(pretrained=True)
    elif sys.argv[1] == "mobilenetv3_small_old":
        model = mobilenetv3_small_old(pretrained=True)
    elif sys.argv[1] == "mobilenetv3_large_old":
        model = mobilenetv3_large_old(pretrained=True)
    else:
        print('''argv[1] must in ["mobilenetv3_large", "moiblenetv3_small", "mobilenetv3_large_old", "mobilenetv3_small_old"]''')
        exit()
    model.eval()
    
    des_path = sys.argv[1] + ".csv"
    deploy_path = sys.argv[1] + ".prototxt"
    model_path = sys.argv[1] + ".caffemodel"
    net = caffe.Net(deploy_path,model_path,caffe.TEST)
   
    src_file = open(src_path)

    data_shape = net.blobs['data'].data.shape
    workers= Pool(4)
    W = 224

    des_file = open(des_path, "w")
    cnt = 0
    idx_begin = 0
    lines = src_file.readlines()
    
    batch_data  = []
    for line in lines:
        line = line.strip()
        ll = line.strip().split(",")
        if len(ll)!=2:
            print("invalid line: " + line)
            continue
        path = ll[0] 
        crop_pct = 0.875 
        img = read_img(path, data_shape[2], crop_pct)
        
        output = model(torch.from_numpy(img).float().unsqueeze(0))[0]
        top5 = output.topk(5)[1].cpu().numpy()
        if int(ll[1]) in top5:
            ptop5_dict['1'] += 1
        else:
            ptop5_dict['0'] += 1
        if int(ll[1]) == top5[0]:
            ptop1_dict['1'] += 1
        else:
            ptop1_dict['0'] += 1
        
        batch_data.append(torch.from_numpy(img).float().unsqueeze(0))
        net.blobs['data'].data[cnt % data_shape[0],...] = img[...]
        
        cnt = cnt + 1
        if cnt % data_shape[0] == 0 or cnt == len(lines):
            net.forward()
            batch_data = []
            for idx_bias in range(cnt - idx_begin):
                top5= np.asarray(net.blobs['fc/fc1'].data[idx_bias]).argsort()[-5:][::-1]
                img_infos = lines[idx_begin + idx_bias].strip().split(",")
                des_file.write(img_infos[0].split('/')[-1] + ", " + img_infos[1] + ", " + \
                    str([net.blobs['fc/fc1'].data[idx_bias][item]  for item in top5]) + '\n')
                if int(img_infos[1]) in top5:
                    top5_dict['1'] += 1
                else:
                    top5_dict['0'] += 1
       
                if int(img_infos[1]) == top5[0]:
                    top1_dict['1'] += 1
                else:
                    top1_dict['0'] += 1
            idx_begin = cnt
        if cnt % 200 == 0:
            print(cnt)
            break
        #if cnt % 1000 == 0:
        #    break
        continue
    des_file.close()

    print("caffe Top-1 accuracy: ", end=' ')
    print(top1_dict['1'] / (top1_dict['1'] + top1_dict['0']))
    
    print("caffe Top-5 accuracy: ", end=' ')
    print(top5_dict['1'] / (top5_dict['1'] + top5_dict['0']))

    print("Pytorch Top-1 accuracy: ", end=' ')
    print(ptop1_dict['1'] / (ptop1_dict['1'] + ptop1_dict['0']))
    
    print("Pytorch Top-5 accuracy: ", end=' ')
    print(ptop5_dict['1'] / (ptop5_dict['1'] + ptop5_dict['0']))
