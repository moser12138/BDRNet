import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict



# from lineResidual import Liner_Risidual
# from lineResidual_slim import Liner_Risidual
# from lineResidual_slim2 import Liner_Risidual
from lineResidual_DRM import Liner_Risidual
# from lineResidual_slim_DRM import Liner_Risidual
# from lineResidual_slim_double_residual import Liner_Risidual


if __name__ == '__main__':

    import time

    device = torch.device('cuda')

    model = Liner_Risidual(n_classes=19, aux_mode='eval')
    model.eval()
    model.to(device)
    iterations = None
    input = torch.randn(1, 3, 1024, 2048).cuda()
    # input = torch.randn(1, 3, 640, 640).cuda()
    # input = torch.randn(1, 3, 960, 720).cuda()
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


