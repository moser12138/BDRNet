import sys
sys.path.insert(0, '.')
import argparse
import torch
import time
from configs import set_cfg_from_file

from lib.models.bisenetv2 import BiSeNetV2
from lib.model_try2.Bisenet_own_gai import BiSeNetV2_own

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,default='configs/cityscapes.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = set_cfg_from_file(args.config)

if __name__ == '__main__':
    args = parse_args()
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')

    # model = BiSeNetV2(n_classes=cfg.n_cats, aux_mode='eval')
    model = BiSeNetV2_own(n_classes=cfg.n_cats, aux_mode='eval')

    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, cfg.eval_crop[0], cfg.eval_crop[1]).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)