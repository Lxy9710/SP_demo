import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry
from predict import prompt_for_sam
from PIL import Image
from utils.utils import cvtColor
import colorsys
import copy
import time

import torch.nn.functional as F
from torch import nn

from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image, show_config
def mix_img(image, pr,count=False, name_classes=None):

    color = np.array([30/255, 144/255, 255/255, 0.6])
    #---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #---------------------------------------------------------#
    image       = cvtColor(image)
    #---------------------------------------------------#
    #   对输入图像进行一个备份，后面用于绘图
    #---------------------------------------------------#
    old_img     = copy.deepcopy(image)
    orininal_h  = np.array(image).shape[0]
    orininal_w  = np.array(image).shape[1]
    #---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    #---------------------------------------------------------#
    image_data, nw, nh  = resize_image(image, (orininal_h, orininal_w ))
    #---------------------------------------------------------#
    #   添加上batch_size维度
    #---------------------------------------------------------#
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

            
    #---------------------------------------------------#
    #   图片传入网络进行预测
    #---------------------------------------------------#

    #---------------------------------------------------#
    #   取出每一个像素点的种类
    #---------------------------------------------------#
    pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
    #--------------------------------------#
    #   将灰条部分截取掉
    #--------------------------------------#
    #---------------------------------------------------#
    #   进行图片的resize
    #---------------------------------------------------#
    pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
    #---------------------------------------------------#
    #   取出每一个像素点的种类
    #---------------------------------------------------#
    pr = pr.argmax(axis=-1)
                # print(seg_img)
    # input('-')

#---------------------------------------------------------#

    seg_img = np.reshape(np.array(color, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
    # print(seg_img)
    # input('-')
    #------------------------------------------------#
    #   将新图片转换成Image的形式
    #------------------------------------------------#
    image   = Image.fromarray(np.uint8(seg_img))
    # print(image)
    # image.save('./out/prompt_model_mask.jpg')
    #------------------------------------------------#
    #   将新图与原图及进行混合
    #------------------------------------------------#
    image   = Image.blend(old_img, image, 0.7)

    
    return image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print(mask_image)
    ax.imshow(mask_image)
    return mask_image
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

if __name__ == "__main__":
#img==your demo pic
    img='img/3.png'
#prompt setup
    input_point,input_label = prompt_for_sam(img)
    input_point= input_point[0:2:]
    input_label=input_label[0:2:]
# --->model
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
#SAM setup
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)


# SAM with prompt
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
#show mask

for i, (mask, score) in enumerate(zip(masks, scores)):
    # Img_mix   = Image.blend(cvtColor(Image.open(img)), mask, 0.7)

    # img_dir='./out/'+'sam_out'+str[i]+'.jpg'
    print(score)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    mask_image=show_mask(mask, plt.gca())
    
    # show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    path='./out/SAM'+str(i)+'.jpg'
    plt.savefig(path)
    plt.axis('off')
    # path='./out/SAM'+str(i)+'.jpg'
    # plt.savefig(path)
    

    # Img_mix   = Image.blend(cvtColor(Image.open(img)), mask_image, 0.7)
    # maskpath='./out/SAMmask'+str(i)+'.jpg'
    # mix=mix_img(Image.open(img),torch.from_numpy(mask))
    # mix_image.save(maskpath)
    plt.show() 
