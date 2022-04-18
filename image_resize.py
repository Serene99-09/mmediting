import os
import cv2

in_folder = "VSR/TheBoyAndTheBeast_Trim_Trim/lq"

for filename in os.listdir(in_folder):
    if filename=='.ipynb_checkpoints' or filename=='.DS_Store':
        continue
    try:
        src = cv2.imread(in_folder + "/" + filename, cv2.IMREAD_UNCHANGED)
        output = cv2.resize(src, (1280,688), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(in_folder + "_upX4/" + filename, output)

    except Exception as e:
        print(filename)
        print(src)
        raise e