from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob


class Sd2_d_multi_feature(object):

    def __init__(self, root, combine_all=True):
        
        self.images_dir = '' 
        self.img_path = osp.join(root) + '/sd2_d_multi_feature'
        self.train_path = self.img_path
        self.gallery_path = ''
        self.query_path = ''
        self.train = []
        self.gallery = []
        self.query = []
        self.num_train_pids = 0
        self.has_time_info = False
        self.load()

    def preprocess(self):
        fpaths = sorted(glob(osp.join(self.images_dir, self.train_path, '*npy')))
        data = []
        all_pids = {}
        
        camid = 0

        for fpath in fpaths:
            fname = osp.basename(fpath)  # filename: 000001_s10_c01_f000295.jpg #299_1.jpg #T000000_1.jpg
            fields = fname.split('_')
            pid = int(fields[0][1:])
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]  # relabel
            data.append((self.img_path + '/' + fname, pid, camid))
            camid = camid + 1
        return data, int(len(all_pids))

    def load(self):
        self.train, self.num_train_pids = self.preprocess()

        print(self.__class__.__name__, "dataset loaded")
        print("  subset | # ids | # images")
        print("  ---------------------------")
        print("  all    | {:5d} | {:8d}"
              .format(self.num_train_pids, len(self.train)))
