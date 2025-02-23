exp_name = 'EXP4_anime_atd12k_selfie_pixar_lr1e-5_400k'
# 'realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost'
scale = 4
# model settings
model = dict(
    type='RealESRGAN',
    generator=dict(
        type='RRDBNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=23,
        growth_channels=32),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=1e-1,
        real_label_val=1.0,
        fake_label_val=0),
    is_use_sharpened_gt_in_pixel=True,
    is_use_sharpened_gt_in_percep=True,
    is_use_sharpened_gt_in_gan=False,
    is_use_ema=True,
)
# model training and testing settings
train_cfg = dict(start_iter=1000000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRWeightDataset'
val_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='Crop', keys=['gt'], crop_size=(400, 400), random_crop=True),
    dict(type='RescaleToZeroOne', keys=['gt']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),
    dict(type='CopyValues', src_keys=['gt_unsharp'], dst_keys=['lq']),
    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 3],
            sigma_y=[0.2, 3],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2]),
        keys=['lq'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.2, 0.7, 0.1],  # up, down, keep
            resize_scale=[0.15, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3.0, 1 / 3.0, 1 / 3.0]),
        keys=['lq'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 30],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 3],
            poisson_gray_noise_prob=0.4),
        keys=['lq'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 95]),
        keys=['lq']),
    dict(
        type='RandomBlur',
        params=dict(
            prob=0.8,
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 1.5],
            sigma_y=[0.2, 1.5],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2]),
        keys=['lq'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.3, 0.4, 0.3],  # up, down, keep
            resize_scale=[0.3, 1.2],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3.0, 1 / 3.0, 1 / 3.0]),
        keys=['lq'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 25],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 2.5],
            poisson_gray_noise_prob=0.4),
        keys=['lq'],
    ),
    dict(
        type='DegradationsWithShuffle',
        degradations=[
            dict(
                type='RandomJPEGCompression',
                params=dict(quality=[5, 50]),
            ),
            [
                dict(
                    type='RandomResize',
                    params=dict(
                        target_size=(100, 100),
                        resize_opt=['bilinear', 'area', 'bicubic'],
                        resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
                ),
                dict(
                    type='RandomBlur',
                    params=dict(
                        prob=0.8,
                        kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
                        kernel_list=['sinc'],
                        kernel_prob=[1],
                        omega=[3.1416 / 3, 3.1416]),
                ),
            ]
        ],
        keys=['lq'],
    ),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(type='Quantize', keys=['lq']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),
    dict(type='ImageToTensor', keys=['lq', 'gt', 'gt_unsharp']),
    dict(
        type='Collect', keys=['lq', 'gt', 'gt_unsharp'], meta_keys=['gt_path'])
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path']),
]
test_pipeline = val_pipeline

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(
        samples_per_gpu=1, drop_last=True,
        persistent_workers=False),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=train_dataset_type,
            lq_folder=['data/atd12k_train/GT', 'data/RealESRGAN_selfie_2048/GT', 'data/RealESRGAN_pixar_2048/GT'],
            gt_folder=['data/atd12k_train/GT', 'data/RealESRGAN_selfie_2048/GT', 'data/RealESRGAN_pixar_2048/GT'],
            ann_file=['data/atd12k_train/meta_info_atd12k.txt', 'data/RealESRGAN_selfie_2048/meta_info_selfie.txt', 'data/RealESRGAN_pixar_2048/meta_info_pixar.txt'],
            pipeline=train_pipeline,
            scale=scale,
            ratio=[0.25, 0.45, 0.3])),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/selfie_test/bicLRx4',
        gt_folder='data/selfie_test/HR',
        pipeline=val_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folder='tmp_frames/org_frames',
        gt_folder='tmp_frames/up_frames', # FAKE gt 
        pipeline=val_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))

# learning policy
total_iters = 20  # 400000
lr_config = dict(policy='Step', by_epoch=False, step=[20], gamma=1)  # 400000

checkpoint_config = dict(interval=20, save_optimizer=True, by_epoch=False) # 5000
evaluation = dict(interval=20, save_image=True) # 5000  (..., gpu_collect=True)
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# custom hook
custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.999),
    )
]

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = 'realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20211010-34798885.pth'  # noqa
resume_from = None
workflow = [('train', 1)]
