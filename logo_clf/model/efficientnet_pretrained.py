from typing import Any
from efficientnet_pytorch import EfficientNet

def efficientnet_b0_pretrained(**kwargs: Any) -> EfficientNet:
    model = EfficientNet.from_pretrained('efficientnet-b0', **kwargs)
    return model

def efficientnet_b1_pretrained(**kwargs: Any) -> EfficientNet:
    model = EfficientNet.from_pretrained('efficientnet-b1', **kwargs)
    return model

def efficientnet_b2_pretrained(**kwargs: Any) -> EfficientNet:
    model = EfficientNet.from_pretrained('efficientnet-b2', **kwargs)
    return model

def efficientnet_b3_pretrained(**kwargs: Any) -> EfficientNet:
    model = EfficientNet.from_pretrained('efficientnet-b3', **kwargs)
    return model

def efficientnet_b4_pretrained(**kwargs: Any) -> EfficientNet:
    model = EfficientNet.from_pretrained('efficientnet-b4', **kwargs)
    return model

def efficientnet_b5_pretrained(**kwargs: Any) -> EfficientNet:
    model = EfficientNet.from_pretrained('efficientnet-b5', **kwargs)
    return model

def efficientnet_b6_pretrained(**kwargs: Any) -> EfficientNet:
    model = EfficientNet.from_pretrained('efficientnet-b6', **kwargs)
    return model

def efficientnet_b7_pretrained(**kwargs: Any) -> EfficientNet:
    model = EfficientNet.from_pretrained('efficientnet-b7', **kwargs)
    return model

def efficientnet_b8_pretrained(**kwargs: Any) -> EfficientNet:
    model = EfficientNet.from_pretrained('efficientnet-b8', **kwargs)
    return model

def efficientnet_l2_pretrained(**kwargs: Any) -> EfficientNet:
    model = EfficientNet.from_pretrained('efficientnet-l2', **kwargs)
    return model


