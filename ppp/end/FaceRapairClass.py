import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer
import warnings

class FRC:

    def __init__(self):

        # params
        version = '1.3'
        upscale = 2
        bg_upsampler = 'realesrgan'

        self.bg_tile = 400
        self.suffix = None
        self.only_center_face = False
        self.aligned = False
        self.ext = 'auto'
        self.weight = 0.5

        if bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():
                print('CUDA недоступна, пропускаем RealESRGAN для фона')
                bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)
        else:
            bg_upsampler = None

        # Settings GFPGAN
        if version == '1':
            arch = 'original'
            channel_multiplier = 1
            model_name = 'GFPGANv1'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        else:
            raise ValueError(f'Неподдерживаемая версия модели: {version}')

        model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = url

        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)

    def pr_face(self, img_path):

        img_name = os.path.basename(img_path)
        print(f'Обрабатываем {img_name}')
        basename, ext_name = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        restorer = self.restorer

        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=self.aligned,
            only_center_face=self.only_center_face,
            paste_back=True,
            weight=self.weight)

        if restored_img is not None:

            if self.ext == 'auto':
                extension = ext_name[1:]
            else:
                extension = self.ext

            img_name_out = f'{basename}_{self.suffix}.{extension}' if self.suffix else f'{basename}.{extension}'
            save_path = os.path.join('results', img_name_out)
            imwrite(restored_img, save_path)
            print(f'Сохранено: {save_path}')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    objj = FRC()
    objj.pr_face('C:/Users/labzi/PycharmProjects/RapairFaceModel/inputs/1.jpg')
