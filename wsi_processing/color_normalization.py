import girder_client
import numpy as np
from skimage.transform import resize
from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
from histomicstk.preprocessing.color_deconvolution.\
    color_deconvolution import color_deconvolution_routine, stain_unmixing_routine
from histomicstk.preprocessing.augmentation.\
    color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration
from PIL import Image
import HistomicsTK_function
import cv2
import glob
import os


source_path = '/media/oasis/41e45ae4-ad45-41db-9bc2-b8299dd12de8/bee/source'
REFER_PATH = '/media/oasis/41e45ae4-ad45-41db-9bc2-b8299dd12de8/bee/reference/1502863-11-HE_40_22000_49000.png'
patch_pathes = glob.glob(os.path.join(source_path, '*.png'))


for patch in patch_pathes:
    source_image = HistomicsTK_function.read_image(patch)
    target_image = HistomicsTK_function.read_reference(REFER_PATH)
    print('source image size: ', source_image.shape)
    print('target image size: ', target_image.shape)
    result = HistomicsTK_function.stain_normalization(source_image, target_image)
    wsi_patch_name = patch.split('/')[-1].split('.png')[0]
    # 保存结果的png文件
    cv2.imwrite(f"/media/oasis/41e45ae4-ad45-41db-9bc2-b8299dd12de8/bee/target/{wsi_patch_name}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


# SOURCE_PATH = '/home/linjianwei/PycharmProjects/Project_LXJ/TCGA_test_output/imgs/svs_patch_10/TCGA-2Y-A9GT-01Z-00-DX1.30666775-3556-4DFE-A5EC-8CCF8EEB1803_10_1792_4704.png'
# TARGET_PATH = '/home/linjianwei/PycharmProjects/Project_LXJ/TCGA_test_output/imgs/svs_patch_10/TCGA-2Y-A9GT-01Z-00-DX1.30666775-3556-4DFE-A5EC-8CCF8EEB1803_10_1568_6272.png'
# RESULT_PATH = '/home/linjianwei/PycharmProjects/Project_LXJ/color_normalization_result'



# #画图  source和target图像
# plt.figure(figsize=(20.0, 20.0))
# plt.subplot(1, 2, 1)
# plt.title('Source', fontsize=20)
# plt.imshow(source_image)
# #以上是soure，以下是target
# plt.subplot(1, 2, 2)
# plt.title('Target', fontsize=20)
# plt.imshow(target_image)
# plt.show()

# #画图 result图像
# plt.figure(figsize=(20.0, 10.0))
# plt.title('Result', fontsize=20)
# plt.imshow(result)
# plt.show()


#画图 source target result三张图
# plt.figure(figsize=(30, 10))
# plt.subplot(1,3,1)
# plt.title('Source', fontsize=50)
# plt.imshow(source_image)
#
# plt.subplot(1,3,2)
# plt.title('Target', fontsize=50)
# plt.imshow(target_image)
#
# plt.subplot(1,3,3)
# plt.title('Result', fontsize=50)
# plt.imshow(result)
# plt.savefig(RESULT_PATH)
# plt.show()


