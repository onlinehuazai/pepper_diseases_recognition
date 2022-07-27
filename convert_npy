import os
import cv2
from os.path import join,dirname,realpath
import numpy as np
import shutil
from argparse import ArgumentParser


if __name__ == "__main__":
    root =  'pepper_diseases/'
    size_new = 512
    save_dir = 'pepper_diseases/npy'

    dir_names=['test'] 
    cnt = 0
    for dir_name in dir_names:
        dir_read = join(root,dir_name)  # 读取目录地址
        dir_save = join(save_dir, dir_name+'_resize{}'.format(size_new))  # 保存地址
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)

        dir_0s = os.listdir(dir_read)
        for dir_0 in dir_0s:

            dir_read_0=join(dir_read,dir_0)
            
            dir_save_0=join(dir_save,dir_0)

            image=cv2.imread(dir_read_0)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            h, w, c = image.shape
            if w > h:
                new_tmp = int(h * size_new * 1.0 / w)
                image_tmp = cv2.resize(image, (size_new, new_tmp))
                image_new = np.zeros((size_new, size_new, 3))
                # image_new[:,:] = np.array([124,116,104]) # imagenet mean
                start = int((size_new-new_tmp)/2)
                image_new[start:start+new_tmp, :, :] = image_tmp
            else:
                new_tmp = int(w * size_new * 1.0 / h)
                image_tmp = cv2.resize(image, (new_tmp, size_new))
                image_new = np.zeros((size_new, size_new, 3))
                # image_new[:,:] = np.array([124,116,104]) # imagenet mean
                start = int((size_new-new_tmp)/2)
                image_new[:, start:start+new_tmp, :] = image_tmp

            # 转化Numpy格式加速IO
            image = image_new.astype(np.uint8)
            image_name = os.path.splitext(dir_0)[0]
            # print(join(dir_save, image_name+'.npy'))
            np.save(join(dir_save, image_name+'.npy'), image)
            cnt = cnt+1
            if cnt % 100 == 0:
                print(cnt)
                
                
'''
import os
import cv2
from os.path import join,dirname,realpath
import numpy as np
import shutil
from argparse import ArgumentParser


if __name__ == "__main__":
    # 保存到同个root下，test_data_A_resize320,train_data_resize320
    # root='/home/LinHonghui/Datasets/HW_ImageRetrieval/'
    root =  'pepper_diseases/'
    size_new = 576
    save_dir = 'pepper_diseases/npy'

    dir_names=['test'] 
    cnt = 0
    for dir_name in dir_names:
        dir_read = join(root,dir_name)  # 读取目录地址
        dir_save = join(save_dir, dir_name+'_resize{}'.format(size_new))  # 保存地址
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)

        dir_0s = os.listdir(dir_read)
        for dir_0 in dir_0s:

            dir_read_0=join(dir_read,dir_0)
            
            dir_save_0=join(dir_save,dir_0)

            image=cv2.imread(dir_read_0)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image_new = cv2.resize(image, (size_new, size_new))
#             h, w, c = image.shape
#             if w > h:
#                 new_tmp = int(h * size_new * 1.0 / w)
#                 image_tmp = cv2.resize(image, (size_new, new_tmp))
#                 image_new = np.zeros((size_new, size_new, 3))
#                 # image_new[:,:] = np.array([124,116,104]) # imagenet mean
#                 start = int((size_new-new_tmp)/2)
#                 image_new[start:start+new_tmp, :, :] = image_tmp
#             else:
#                 new_tmp = int(w * size_new * 1.0 / h)
#                 image_tmp = cv2.resize(image, (new_tmp, size_new))
#                 image_new = np.zeros((size_new, size_new, 3))
#                 # image_new[:,:] = np.array([124,116,104]) # imagenet mean
#                 start = int((size_new-new_tmp)/2)
#                 image_new[:, start:start+new_tmp, :] = image_tmp

            # 转化Numpy格式加速IO
            image = image_new.astype(np.uint8)
            image_name = os.path.splitext(dir_0)[0]
            # print(join(dir_save, image_name+'.npy'))
            np.save(join(dir_save, image_name+'.npy'), image)
            cnt = cnt+1
            if cnt % 100 == 0:
                print(cnt)
'''
