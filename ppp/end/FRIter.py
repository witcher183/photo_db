import os
import time
import warnings
from FaceRapairClass import FRC

warnings.filterwarnings('ignore')
worker = FRC()

while True:

    if len(os.listdir('inputs')) > 0:

        images_arr = os.listdir('inputs')

        for path in images_arr:

            worker.pr_face(os.path.join('inputs', path))
            os.remove(os.path.join('inputs', path))

            print(f'Изображение {path} удалено')

    else:

        time.sleep(5)
