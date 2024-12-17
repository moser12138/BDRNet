
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=19,
    num_aux_heads=4,
    # lr_start=5e-3,
    lr_start=0.005,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=300000,
    # max_iter=150000,
    dataset='CityScapes',
    im_root='./datasets/cityscapes',
    train_im_anns='./datasets/cityscapes/train.txt',
    val_im_anns='./datasets/cityscapes/val.txt',
    # scales=[0.25, 2.],
    scales=[0.5, 2.],
    # cropsize=[512, 1024],
    cropsize=[1024, 1024],
    eval_crop=[1024, 1024],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    # ims_per_gpu=17,
    # ims_per_gpu=2,
    ims_per_gpu=16,
    # ims_per_gpu=4,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res',
)