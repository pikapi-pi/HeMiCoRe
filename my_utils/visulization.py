import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
import pandas as pd
from fuzzywuzzy import fuzz, process
import os, csv, json
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
from dhg.random import hypergraph_Gnm
import dhg
import networkx as nx
from utils import get_three_points, keep_patch, out_of_bound
import openslide
from collections import Counter
from io import StringIO
import random, matplotlib



def key_patches(patch_size, magnification, case_id=None, label=None, overview_level=None):
    for state in ['overall', 'original', 'mid', 'final']:
        if state != "overall":
            key_patches_path = f'../results/HCC/key_patch/{state}/key_patches_on_feature.json'
            with open(key_patches_path, 'r') as f:
                all_data = json.load(f)
            print(all_data.keys())
            if case_id in all_data.keys():
                print('OK')
            if state != 'original':
                case_final_key_patchs = all_data[case_id]['5']
            else:
                case_final_key_patchs = all_data[case_id]['0']
        slide_path = f'/media/oasis/DATA/survival_prediction/data/{case_id}.svs'
        details_path = f'/media/oasis/DATA/survival_prediction/new_patch/coord/{case_id}.json'
        all_patch_cluster_detail_path = f"/media/oasis/DATA/survival_prediction/new_patch/features/resnet18/k-means-10/{case_id}.json"
        with open(details_path, 'r') as fp:
            data = json.load(fp)
        coords = data['coords']
        with open(all_patch_cluster_detail_path, 'r') as f1:
            cluster_data = json.load(f1)



        slide = openslide.open_slide(str(slide_path))
        if 'aperio.AppMag' in slide.properties.keys():
            level0_magnification = int(slide.properties['aperio.AppMag'])
        elif 'openslide.mpp-x' in slide.properties.keys():
            level0_magnification = 40 if int(np.floor(float(slide.properties['openslide.mpp-x']) * 10)) == 2 else 20
        else:
            level0_magnification = 40

        patch_size_level0 = int(patch_size * (level0_magnification / magnification))
        thumbnail = slide.get_thumbnail(slide.level_dimensions[overview_level]).convert('RGB')
        thumbnail_1 = slide.get_thumbnail(slide.level_dimensions[overview_level]).convert('RGB')
        thumbnail_2 = slide.get_thumbnail(slide.level_dimensions[overview_level]).convert('RGB')
        thumbnail = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2BGR)
        thumbnail_1 = cv2.cvtColor(np.asarray(thumbnail_1), cv2.COLOR_RGB2BGR)
        thumbnail_2 = cv2.cvtColor(np.asarray(thumbnail_2), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f'/media/oasis/DATA/survival_prediction/figs/key_patch_visulization/HCC/{state}/{label}/{case_id}_raw.png', thumbnail)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 128),
                  (65, 105, 225), (255, 165, 0), (128, 0, 128)]
        # visualizing the distribution maps of candidate patches
        if state == "overall":
            for i in range(len(cluster_data)):
                for index in cluster_data[i]:
                    points_thumbnail = get_three_points(coords[index]['col'], coords[index]['row'],
                                                        patch_size_level0 / slide.level_downsamples[overview_level])
                    cv2.rectangle(thumbnail_2, points_thumbnail[0], points_thumbnail[1], color=colors[i], thickness=-1)
            cv2.imwrite(
                f'../survival_prediction/figs/key_patch_visulization/HCC/{state}/{label}/{case_id}.png',
                thumbnail_2)
        else:
            # visualizing the distribution maps of selected representative patches
            for i in range(len(case_final_key_patchs)-1):#11条超边，最后一条联系前十条超边
                flag = True
                for index in case_final_key_patchs[i]:
                    # whether to memory the examples of each cluster
                    if flag:
                        patch_level0 = slide.read_region(location=(coords[index]['x'], coords[index]['y']), level=0,
                                                         size=(patch_size_level0, patch_size_level0)).convert('RGB')
                        patch = patch_level0.resize(size=(patch_size, patch_size))
                        patch.save(f'/media/oasis/DATA/survival_prediction/figs/key_patch_visulization/CPS/{label}/{i}.png')
                    flag = False
                    points_thumbnail = get_three_points(coords[index]['col'], coords[index]['row'],
                                                        patch_size_level0 / slide.level_downsamples[overview_level])
                    cv2.rectangle(thumbnail, points_thumbnail[0], points_thumbnail[1], color=colors[i], thickness=5)
                    cv2.rectangle(thumbnail_1, points_thumbnail[0], points_thumbnail[1], color=colors[i], thickness=-1)
            if state != "original":
                cv2.imwrite(f'/media/oasis/DATA/survival_prediction/figs/key_patch_visulization/HCC/{state}/{label}/{case_id}_5_{state}_{label}.png', thumbnail)
                cv2.imwrite(
                    f'/media/oasis/DATA/survival_prediction/figs/key_patch_visulization/HCC/{state}/{label}/{case_id}_5(1)_{state}_{label}.png',
                    thumbnail_1)
            else:
                cv2.imwrite(
                    f'/media/oasis/DATA/survival_prediction/figs/key_patch_visulization/HCC/{state}/{label}/{case_id}_0_{state}_{label}.png',
                    thumbnail)
                cv2.imwrite(
                    f'/media/oasis/DATA/survival_prediction/figs/key_patch_visulization/HCC/{state}/{label}/{case_id}_0(1)_{state}_{label}.png',
                    thumbnail_1)

def run_main():

    key_patches(patch_size=256, magnification=20, case_id="1715836-HE", label='negative', overview_level=2)
    # key_patches_at_random(patch_size=256, magnification=20, case_id="2010531-HE", state='original', label='negative', overview_level=2)
    # structure_visulization(space="feature", case_id="TCGA-51-4080-01Z-00-DX1.fd660ce3-3e30-4907-b54a-81002ec071f2")


if __name__ == '__main__':

    run_main()