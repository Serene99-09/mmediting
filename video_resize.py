'''
This scripts help to resize the input video and save the frames.
'''

import os
import cv2

videoCapture = cv2.VideoCapture('VSR/SpiritedAway_Trim/SpiritedAway_Trim.mp4')

fps = 30  # 保存视频的帧率
size = (480, 260)  # 保存视频的大小

videoWriter = cv2.VideoWriter('VSR/SpiritedAway_Trim/SpiritedAway_Trim_downX4.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
i = 0

while True:
    success, frame = videoCapture.read()
    if success:
        i += 1
        if (i >= 1 and i <= 8000):
            # save the original video to images
            cv2.imwrite('VSR/SpiritedAway_Trim/gt/'+'0'*(8-len(str(i-1)))+str(i-1)+'.png', frame)
            # save the new video to images
            frame = cv2.resize(frame, size)
            cv2.imwrite('VSR/SpiritedAway_Trim/lq/'+'0'*(8-len(str(i-1)))+str(i-1)+'.png', frame)
            # save the new video to video
            videoWriter.write(frame)

        if (i > 8000):
            print("success resize")
            break
    else:
        print('end')
        break


# resize the lq as gt
in_folder = "VSR/SpiritedAway_Trim/lq"

for filename in os.listdir(in_folder):
    if filename=='.ipynb_checkpoints' or filename=='.DS_Store':
        continue
    try:
        src = cv2.imread(in_folder + "/" + filename, cv2.IMREAD_UNCHANGED)
        output = cv2.resize(src, (1920,1040), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(in_folder + "_upX4/" + filename, output)

    except Exception as e:
        print(filename)
        print(src)
        raise e