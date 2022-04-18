import copy
import numpy as np
import os.path as osp
import random

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS
import pdb

@DATASETS.register_module()
class SRGLEANDataset(BaseSRDataset):

    def __init__(self,
                 gpen_lq_folder,
                 gpen_gt_folder,
                 ori_gt_folder,
                 pipeline,
                 scale,
                 gpen_ratio=0.5,
                 test_mode=False,
                 filename_tmpl='{}'):

        super().__init__(pipeline, scale, test_mode)

        if isinstance(gpen_lq_folder, list):
            assert(len(gpen_lq_folder)==len(gpen_gt_folder)==len(gpen_ratio))
            self.gpen_ratio = gpen_ratio
            self.gpen_lq_folder =  gpen_lq_folder
            self.gpen_gt_folder = gpen_gt_folder
        else:
            # assert(isinstance(self.gpen_lq_folder, str))
            # assert(isinstance(self.gpen_gt_folder, str))
            # assert(isinstance(self.gpen_ratio, float))
            self.gpen_ratio = [gpen_ratio]
            self.gpen_lq_folder =  [gpen_lq_folder]
            self.gpen_gt_folder = [gpen_gt_folder]
        # compute orig ratio

        self.ori_gt_folder = str(ori_gt_folder)
        
        self.orig_ratio = 1.0
        for fi in range(len(self.gpen_ratio)):
            self.orig_ratio -= self.gpen_ratio[fi]
        assert(self.orig_ratio>=0)
        self.all_ratio= [self.orig_ratio]
        self.all_ratio.extend(self.gpen_ratio)

        

        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

        self.id_start = [0]
        id_start =0 
        for fi in range(len(self.num_all)):
            id_start += self.num_all[fi]
            self.id_start.append(id_start)


    def get_data_info(self, gt_folder, lq_folder=None):
        data_infos = []

        gt_paths = sorted(self.scan_folder(gt_folder))
        lq_paths = None if lq_folder is None else sorted(
            self.scan_folder(lq_folder))

        for i, gt_path in enumerate(gt_paths):
            if lq_paths is not None:
                lq_path = lq_paths[i]
                data_infos.append(dict(lq_path=lq_path, gt_path=gt_path))
            else:
                data_infos.append(dict(lq_path=None, gt_path=gt_path))

        return data_infos

    def load_annotations(self):
        """Load annoations for SR dataset.

        It loads the LQ and GT image path from folders.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """

        data_infos_ori = self.get_data_info(self.ori_gt_folder, None)

        self.num_ori = len(data_infos_ori)

        data_infos_gpen = []
        self.num_all = [self.num_ori]
        for fi in range(len(self.gpen_lq_folder)):
       
            data_infos_gpen_fi = self.get_data_info(str(self.gpen_gt_folder[fi]),
                                str(self.gpen_lq_folder[fi]))
            data_infos_gpen.extend(data_infos_gpen_fi)
            self.num_all.append(len(data_infos_gpen_fi)) 

        return data_infos_ori + data_infos_gpen

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """

        # random select a dataset idx
        fidx = np.random.choice(np.arange(0,len(self.all_ratio)), p=self.all_ratio)
        idx = random.randint(self.id_start[fidx], self.id_start[fidx+1]-1) #Return random integer in range [a, b], including both end points.


        results = copy.deepcopy(self.data_infos[idx])
        results['scale'] = self.scale
        return self.pipeline(results)
