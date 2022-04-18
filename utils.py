import os
import os.path as osp

in_folder = "selfie2anime_RE"
for filename in os.listdir(in_folder):
    basename, ext = osp.splitext(osp.basename(filename))
    if len(basename) == 4:
        print(filename)
        cmd = f'mv selfie2anime_RE/{filename} selfie_val_RE'
        os.system(cmd)
