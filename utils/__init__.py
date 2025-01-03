from .train import train
# from .test import test
# from .infer import predict
from .dataloader import download_mnist

__all__ = [
    'train',
    # 'test',
    # 'predict',
    'download_mnist'
] 