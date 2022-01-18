from pytorch_pretrained_vit import ViT


def init_vit_from_config(config):
    model_name = config.pop('model_name')
    model = Vit(model_name, pretrained=True, **config)
    return model

def vit_b16(**kwargs):
    model = ViT('B_16', pretrained=True, **kwargs)
    return model

def vit_b32(**kwargs):
    model = ViT('B_32', pretrained=True, **kwargs)
    return model

def vit_l16(**kwargs):
    model = ViT('L_16', pretrained=True, **kwargs)
    return model

def vit_l32(**kwargs):
    model = ViT('L_32', pretrained=True, **kwargs)
    return model

def vit_b16_imagenet1k(**kwargs):
    model = ViT('B_16_imagenet1k', pretrained=True, **kwargs)
    return model

def vit_b32_imagenet1k(**kwargs):
    model = ViT('B_32_imagenet1k', pretrained=True, **kwargs)
    return model

def vit_l16_imagenet1k(**kwargs):
    model = ViT('L_16_imagenet1k', pretrained=True, **kwargs)
    return model

def vit_l32_imagenet1k(**kwargs):
    model = ViT('L_32_imagenet1k', pretrained=True, **kwargs)
    return model
