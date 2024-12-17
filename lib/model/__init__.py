# from .lineResidual import Liner_Risidual
from .lineResidual_DRM import Liner_Risidual
# from .lineResidual_DRM_no_liner_compare import Liner_Risidual
# from .lineResidual_slim import Liner_Risidual
# from .lineResidual_slim2 import Liner_Risidual
# from .lineResidual_slim_nores import Liner_Risidual
# from .lineResidual_slim_DRM import Liner_Risidual
# from .lineResidual_slim_double_residual import Liner_Risidual
# from .DDRNet_23_slim import DualResNet
# from .lineResidual_slim_DRM_attention import Liner_Risidual


model_factory = {
    'bisenetv2': Liner_Risidual,
    # 'bisenetv2': DualResNet,
}