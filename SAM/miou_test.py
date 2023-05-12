import random
import os

import cv2
import numpy as np
from tqdm import tqdm

from segment_anything import build_sam, SamAutomaticMaskGenerator

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 模型加载
mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint='sam_vit_h_4b8939.pth'))

def generate_seg_result(image_path, save_path):
    """ SAM模型生成分割标签

    Args:
        image_path (str): 待分割图像路径
        save_path (str): 保存图像路径
    """
    # 3090测试推理时间为100 ms 后处理为100 ms
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    # 船端模型
    
    # 如果船端模型检测到有水域，使用SAM模型进行推理
    # SAM模型
    masks = mask_generator.generate(image.copy())
    # print(masks)
    # input('--------------------mask check------------')
    for result in masks:
        x1, y1, xw, xh = result['bbox']
        # 根据船端模型找到大致水的区域，sum(mapA * mapB) > 0.8 
        if result['area'] > 104650 and (y1 + xh/2) > h*2/3:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            print(x1, y1, xw, xh)
            mask_image = np.zeros((h, w))
            mask_image[result['segmentation']] = 255
            # 对左侧和右侧进行横向膨胀
            # mask_image[:, :10] = cv2.dilate(mask_image[:, :10], np.ones((1, 5), np.uint8), iterations = 1)
            # mask_image[:, -10:] = cv2.dilate(mask_image[:, -10:],  np.ones((1, 5), np.uint8), iterations = 1)
            contours, _ = cv2.findContours(mask_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image, contours, -1, (b, g, r), -1)
            
    cv2.imwrite(save_path, image)

if __name__ == "__main__":
    image_root = 'fulldataset/JPEGImages'
    with open(os.path.join('robosam/SAM/fulldataset/ImageSets/Segmentation/val.txt'),"r") as f:
        val_lines = f.readlines()
    name            = val_lines.split()[0]
    print(name)
    # data_list = os.listdir(image_root)
    # data_list.sort()
    # for i in tqdm(range(0, len(data_list))):
    #     # print(data_list[i])
    #     # input('---')
    #     image_path = image_root + data_list[i]
    #     save_path = './save/' +  data_list[i]
    #     if not os.path.exists('./save/'):
    #         os.mkdir('./save/')
    #     generate_seg_result(image_path, save_path)