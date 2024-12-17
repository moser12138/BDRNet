#train
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/cityscapes.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train.py --config $cfg_file

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/camvid.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train.py --config $cfg_file

export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/coco.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train.py --config $cfg_file

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/ade20k.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train.py --config $cfg_file
