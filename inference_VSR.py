import os
import argparse
import glob
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='Nezha_input_480_Trim.mp4', help='Input video')
    ################ args for tools/test.py ################
    parser.add_argument('--config', default='configs/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_anime.py', help='test config file path')
    parser.add_argument('--checkpoint', default='anime_paths/EXP1/iter_100000.pth', help='checkpoint file')
    parser.add_argument(
        '--save-path',
        default='tmp_frames/inference_frames',
        type=str,
        help='path to store images and if not given, will not save image')
    ################################################################
    parser.add_argument('--fps', type=float, default=30, help='FPS of the output video')
    parser.add_argument('-o', '--output', type=str, default='crayon', help='Folder to store the output video')
    args = parser.parse_args()
    
    # Step 1: Prepare image folders to be inferenced from the input video
    video_name = os.path.splitext(os.path.basename(args.input))[0]

    # 1. LQ folder: tmp_frames/org_frames
    # use ffmpeg to extract frames from input video
    frame_folder = os.path.join('tmp_frames', 'org_frames')
    os.makedirs(frame_folder, exist_ok=True)
    os.system(f'ffmpeg -i {args.input} -qscale:v 1 -qmin 1 -qmax 1 -vsync passthrough  {frame_folder}/frame%08d.png')
    
    # obtain the original resolution
    img_path = os.listdir(frame_folder)[0]
    src = cv2.imread(frame_folder + "/" + img_path, cv2.IMREAD_UNCHANGED)
    
    # 2. GT folder: tmp_frames/up_frames
    # Generate fake GT folder since we are using SRFolderDataset in the config file
    # To generate fake GT video by directly upsampling by 4
    # os.system(f'ffmpeg -i {args.input} -vf scale=1920:1040 {video_name}_X4.mp4')
    os.system(f'ffmpeg -i {args.input} -vf scale={src.shape[1]*4}:{src.shape[0]*4} {video_name}_X4.mp4')
    # use ffmpeg to extract frames from upsampled video
    frame_folder = os.path.join('tmp_frames', 'up_frames')
    os.makedirs(frame_folder, exist_ok=True)
    os.system(f'ffmpeg -i {video_name}_X4.mp4 -qscale:v 1 -qmin 1 -qmax 1 -vsync passthrough  {frame_folder}/frame%08d.png')

    # Step 2: Make inference using tools/test.py
    # os.system(f'python tools/test.py --config {args.config} --checkpoint {args.checkpoint} --save-path {args.save_path}')

    # # Step 3: Merge output frames (stored at tmp_frames/inference_frames by default) as output video
    # # get input video fps if not specified
    # # if args.fps is None:
    # #     import ffmpeg
    # #     probe = ffmpeg.probe(args.input)
    # #     video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    # #     args.fps = eval(video_streams[0]['avg_frame_rate'])

    # video_save_path = os.path.join(args.output, f'{video_name}_out.mp4')
    # os.makedirs(args.output, exist_ok=True)
    # os.system(f'ffmpeg -r {args.fps} -i {args.save_path}/frame%08d.png '
    #                   f'-c:v libx264 -r {args.fps} -pix_fmt yuv420p {video_save_path}')

if __name__ == '__main__':
    main()
