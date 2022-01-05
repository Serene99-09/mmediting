import cv2

videoCapture = cv2.VideoCapture('VSR/TheBoyAndTheBeast_Trim_Trim.mp4')

fps = 30  # 保存视频的帧率
size = (320, 172)  # 保存视频的大小

videoWriter = cv2.VideoWriter('VSR/TheBoyAndTheBeast_Trim_Trim_downX4.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
i = 0

while True:
    success, frame = videoCapture.read()
    if success:
        i += 1
        if (i >= 1 and i <= 8000):
            # save the original video to images
            cv2.imwrite('VSR/TheBoyAndTheBeast_Trim_Trim/gt/frame_'+str(i)+'.png', frame)
            # save the new video to images
            frame = cv2.resize(frame, (320, 172))
            cv2.imwrite('VSR/TheBoyAndTheBeast_Trim_Trim/lq/frame_'+str(i)+'.png', frame)
            # save the new video to video
            videoWriter.write(frame)

        if (i > 8000):
            print("success resize")
            break
    else:
        print('end')
        break
