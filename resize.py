import os
import cv2

src = cv2.imread("tmp/org/18.jpg", cv2.IMREAD_UNCHANGED)
output = cv2.resize(src, (int(src.shape[1]*4),int(src.shape[0]*4)), interpolation = cv2.INTER_CUBIC)
cv2.imwrite("tmp/up/18.jpg", output)


# in_folder = "selfie2anime_256"
# for filename in os.listdir(in_folder):
#     src = cv2.imread(in_folder + "/" + filename, cv2.IMREAD_UNCHANGED)
#     output = cv2.resize(src, (1024, 1024), interpolation = cv2.INTER_CUBIC)
#     cv2.imwrite(in_folder[:-3] + "1024/" + filename, output)

# bad_cnt = 0
# in_folder = "imagenette2-160/val"
# for filename in os.listdir(in_folder):
#     src = cv2.imread(in_folder + "/" + filename, cv2.IMREAD_UNCHANGED)
#     #src = read_image(in_folder + '/' + filename)
#     #write_png(src,'test.png')
#     if len(src.shape)!=3:
#         bad_cnt+=1
#         print(filename)
#         cmd = f'rm {in_folder}/{filename}'
#         os.system(cmd)
#     else:
#         try: 
#             output = cv2.resize(src, (int(src.shape[1]*4),int(src.shape[0]*4)), interpolation = cv2.INTER_CUBIC)
#             cv2.imwrite("imagenette2-160/valX4" + "/" + filename, output)
#         except:
#             print(filename)

# print(bad_cnt)


        