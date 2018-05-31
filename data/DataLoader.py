import os
import sys
sys.path.append('../')
import cfg
import glob
import h5py
import numpy as np
from tqdm import tqdm

class DataLoader(object):
    def __init__(self, image_path, preprocessed_path):
        self.image_path = image_path

        self.all_image_dir = glob.glob(os.path.join(self.image_path, 'n*'))
        self.all_image_dir = np.sort(self.all_image_dir)
        self.all_dirs = {}

        for each_image_dir in tqdm(self.all_image_dir):
            each_image_dir_name = each_image_dir.split('/')[-1]
            if not each_image_dir_name in self.all_dirs:
                self.all_dirs[each_image_dir_name] = [] 
            all_dir_image = glob.glob(os.path.join(each_image_dir, '*.JPEG'))
            all_dir_image = np.sort(all_dir_image)
            for each_dir_image in all_dir_image:
                # print each_dir_image
                self.all_dirs[each_image_dir_name].append(each_dir_image)

        self.preprocessed_path = preprocessed_path
        for each_image_dir in tqdm(self.all_image_dir):
            h5_path = os.path.join(each_image_dir, '.h5')
            if os.path.exists(h5_path):
               continue
               print 'Skip store image class %s'%(each_image_dir.split('/')[-1])
            else:
                h5_data = h5py.File(h5_path)

                h5_data.close()
            break


if __name__ == '__main__':
    data_loader = DataLoader(image_path=cfg.IMAGENET_PATH,
                             preprocessed_path='./preprocessed/Imagenet')
    main()