import cv2
import numpy as np
import histomicstk.preprocessing.color_normalization.deconvolution_based_normalization as deconvolution_based_normalization



def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv default color space is BGR, change it to RGB
    # img=np.array(img)
    p = np.percentile(img, 90)
    img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)

    return img

def read_reference(path):
    target = cv2.imread(path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB) # opencv default color space is BGR, change it to RGB
    # img=np.array(img)
    p = np.percentile(target, 90)
    target = np.clip(target * 255.0 / p, 0, 255).astype(np.uint8)

    return target



def stain_normalization(img,target):
   stain_unmixing_routine_params = {
        'stains': ['hematoxylin','eosin'],
        'stain_unmixing_method':'macenko_pca',

    }

   tissue_rgb_normalized = deconvolution_based_normalization(
       img,
       im_target=target,
       stain_unmixing_routine_params=stain_unmixing_routine_params
   )

   return tissue_rgb_normalized
