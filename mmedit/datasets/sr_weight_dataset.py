# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import os.path as osp
import random

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS
import pdb
 
@DATASETS.register_module()
class SRWeightDataset(BaseSRDataset):
    """General paired image dataset with an annotation file for image
    restoration. 

    Special thing here is that the train dataset is a combination of several
    gt sets, and each of them has a weight sepcified in the config file.

    The dataset loads lq (Low Quality) and gt (Ground-Truth) image pairs,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "annotation file mode":
    Each line in the annotation file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an annotation file:

    ::

        0001_s001.png (480,480,3)
        0001_s002.png (480,480,3)

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
    """
    def __init__(self,
                 lq_folder,
                 gt_folder,
                 ann_file,
                 pipeline,
                 scale,
                 ratio=1.0,
                 test_mode=False,
                 filename_tmpl='{}'):

        super().__init__(pipeline, scale, test_mode)

        if isinstance(lq_folder, list):
            assert(len(lq_folder)==len(gt_folder)==len(ratio)==len(ann_file))
            self.ratio = ratio
            self.ann_file = ann_file
            self.lq_folder =  lq_folder
            self.gt_folder = gt_folder
        else:
            # assert(isinstance(self.gpen_lq_folder, str))
            # assert(isinstance(self.gpen_gt_folder, str))
            # assert(isinstance(self.gpen_ratio, float))
            self.ratio = [ratio]
            self.ann_file = [ann_file]
            self.lq_folder =  [lq_folder]
            self.gt_folder = [gt_folder]
        
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

        self.id_start = [0]
        id_start =0 
        for fi in range(len(self.num_all)):
            id_start += self.num_all[fi]
            self.id_start.append(id_start)

    def get_data_info(self, ann_file, gt_folder, lq_folder):
        data_infos = []
        with open(ann_file, 'r') as fin:
            for line in fin:
                gt_name = line.split(' ')[0]
                basename, ext = osp.splitext(osp.basename(gt_name))
                lq_name = f'{self.filename_tmpl.format(basename)}{ext}'
                data_infos.append(
                    dict(
                        lq_path=osp.join(lq_folder, lq_name),
                        gt_path=osp.join(gt_folder, gt_name)))
        return data_infos

    def load_annotations(self):
        """Load annoations for SR dataset.

        It loads the LQ and GT image path from folders.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """

        data_infos = []
        self.num_all = []
        for fi in range(len(self.lq_folder)):
            data_infos_fi = self.get_data_info(str(self.ann_file[fi]), 
                            str(self.gt_folder[fi]), str(self.lq_folder[fi]))
            data_infos.extend(data_infos_fi)
            self.num_all.append(len(data_infos_fi)) 
        return data_infos                   
        
    def __getitem__(self, idx):
        """(Overwrite that in BaseSRDataset) Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """

        # randomly select a dataset idx
        fidx = np.random.choice(np.arange(0,len(self.ratio)), p=self.ratio)
        idx = random.randint(self.id_start[fidx], self.id_start[fidx+1]-1) #Return random integer in range [a, b], including both end points.


        results = copy.deepcopy(self.data_infos[idx])
        results['scale'] = self.scale
        return self.pipeline(results)
