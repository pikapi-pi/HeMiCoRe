import os
import json
import argparse
import time

import openslide
import numpy as np
from tqdm import tqdm
from pathlib import Path
# import HistomicsTK_function
import torch,sys
from torch import nn
import torchvision.transforms as torch_trans
from torchvision import transforms
from torchvision import models
from PIL import Image

from ..stain_normalization import stainNorm_Reinhard, stainNorm_Vahadane, stainNorm_Macenko
import cv2
from torch.multiprocessing import Process, set_start_method


def stain_normalization(img, REFER_PATH, c, args, encoder):
    cd = np.array([c['row'], c['col']], dtype=np.int32)
    refer_img = cv2.cvtColor(cv2.imread(REFER_PATH), cv2.COLOR_BGR2RGB)
    img = np.asarray(img)
    try:
        n = stainNorm_Reinhard.Normalizer()
        n.fit(refer_img)
        out = n.transform(img)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        out = Image.fromarray(out)
        feature = extract(args, out, encoder, transform)
        return feature, cd
    except:
        return None, np.array([-1, -1])

def create_encoder(args):
    print(f"Info: Creating extractor {args.image_encoder}")
    if args.image_encoder == 'vgg16':
        encoder = models.vgg16(pretrained=True).to(args.device)
        encoder.classifier = nn.Sequential(*list(encoder.classifier.children())[:-3])
    elif args.image_encoder == 'resnet50':
        encoder = models.resnet50(pretrained=True).to(args.device)
        layers = list(encoder.children())[:-1]
        layers.append(nn.Flatten(1))
        encoder = nn.Sequential(*layers)
    elif args.image_encoder == 'resnet18':
        encoder = models.resnet18(pretrained=True).to(args.device)
        layers = list(encoder.children())[:-1]
        layers.append(nn.Flatten(1))
        encoder = nn.Sequential(*layers)
    elif args.image_encoder == 'densenet':
        encoder = models.densenet121(pretrained=True).to(args.device)
        print("OK")
        for param in encoder.parameters():
            param.requires_grad = False
        encoder = nn.Sequential(encoder.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        layers = list(encoder.children())
        layers.append(nn.Flatten(1))
        encoder = nn.Sequential(*layers)
        encoder = nn.DataParallel(encoder)
        checkpoint = torch.load('../KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth')
        del checkpoint["module.fc_4.weight"]
        del checkpoint["module.fc_4.bias"]
        encoder.load_state_dict({k.replace('module.model', 'module'): v for k, v in checkpoint.items()}, strict=True)
    else:
        raise ValueError(f"image_encoder's name error!")
    print(f"{args.image_encoder}:\n{encoder}")
    return encoder


def extract(args, image, encoder, transform=None):
    with torch.no_grad():
        if transform is None:
            to_tensor = torch_trans.ToTensor()
            image = to_tensor(image).unsqueeze(dim=0).to(args.device)
        else:
            image = transform(image).unsqueeze(dim=0).to(args.device)
        feat = encoder(image).cpu().numpy()
        return feat


def extract_features(args, encoder, save_dir):
    # get `coord` files
    coord_dir = Path(args.patch_dir) / args.coord
    if not coord_dir.exists():
        print(f"{str(coord_dir)} doesn't exist! ")
        return
    coord_list = sorted(list(coord_dir.glob('*.json')))
    print(f"num of coord: {len(coord_list)}")

    with torch.no_grad():
        encoder.eval()
        for i, coord_filepath in enumerate(coord_list):
            filename = coord_filepath.stem
            npz_filepath = save_dir / f'{filename}.npz'
            if npz_filepath.exists() and not args.exist_ok:
                print(f"{npz_filepath.name} is already exist, skip!")
                continue

            # obtain the parameters needed for feature extraction from `coord` file
            start = time.time()
            with open(coord_filepath) as fp:
                coord_dict = json.load(fp)
            num_patches = coord_dict['num_patches']
            if num_patches == 0:
                print(f"{filename}'s num_patches is {num_patches}, skip!")
                continue
            num_row, num_col = coord_dict['num_row'], coord_dict['num_col']
            coords = coord_dict['coords']
            patch_size_level0 = coord_dict['patch_size_level0']
            patch_size = coord_dict['patch_size']
            slide = openslide.open_slide(coord_dict['slide_filepath'])

            coords_bar = tqdm(coords)
            features, cds = [], []
            REFER_PATH = args.refer_img

            results = []
            ctx = torch.multiprocessing.get_context("spawn")
            pool = ctx.Pool(args.process)
            for c in coords_bar:
                img = slide.read_region(
                    location=(c['x'], c['y']),
                    level=0,
                    size=(patch_size_level0, patch_size_level0)
                ).convert('RGB').resize((patch_size, patch_size))
                results.append(pool.apply_async(stain_normalization, (img, REFER_PATH, c, args, encoder)))
                coords_bar.set_description(
                    f"{i + 1:3}/{len(coord_list):3} | filename: {filename}")
                coords_bar.update()
            pool.close()
            pool.join()
            for result in results:
                feature, cd = result.get()
                flag = np.array([-1, -1])
                if any(cd != flag):
                    features.append(feature)
                    cds.append(cd)
            # for c in coords_bar:
            #     img = slide.read_region(
            #             location=(c['x'], c['y']),
            #             level=0,
            #             size=(patch_size_level0, patch_size_level0)
            #         ).convert('RGB').resize((patch_size, patch_size))
            #     feature, cd = stain_normalization(img, REFER_PATH, c, args, encoder)
            #     coords_bar.set_description(
            #         f"{i + 1:3}/{len(coord_list):3} | filename: {filename}")
            #     coords_bar.update()
            #     features.append(feature)
            #     cds.append(cd)
            print("extract done!")
            img_features = np.concatenate(features, axis=0)
            cds = np.stack(cds, axis=0)
            # save all the patch features of a WSI as a .npz file
            np.savez(file=npz_filepath,
                     filename=filename,
                     num_patches=num_patches,
                     num_row=num_row,
                     num_col=num_col,
                     img_features=img_features,
                     coords=cds)
            end = time.time()
            print(f"用时{(end-start)/60}分")


def run(args):
    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("OK")
    else:
        args.device = torch.device('cpu')

    # Save Directory
    if args.save_dir is not None:
        save_dir = Path(args.save_dir) / args.image_encoder
    else:  # if the save directory is not specified, `patch_dir/features/${image_encoder}` is used by default
        save_dir = Path(args.patch_dir) / 'features' / args.image_encoder
    save_dir.mkdir(parents=True, exist_ok=True)

    if Path(save_dir).exists():
        print(f"{save_dir} is already exists. ")

    encoder = create_encoder(args)
    extract_features(args, encoder, save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, default='', help='Directory containing `coord` files')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--image_encoder', type=str, default='resnet18')
    parser.add_argument('--device', default='0')
    parser.add_argument('--process', default=16, help='the number of processors')
    parser.add_argument('--exist_ok', action='store_true', default=False)
    parser.add_argument('--coord', default='coord', help='for multiprocceing')
    parser.add_argument('--refer_img', default='./12_44.png', help='target image for stain_normalization')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
