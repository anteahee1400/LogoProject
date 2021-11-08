# https://github.com/utkuozbulak/pytorch-cnn-visualizations

import PIL
from PIL import Image
import cv2
import numpy as np

from torchvision import transforms
import torch
import torch.nn as nn


class CamExtractor:
    def __init__(self, model, target_layer, fc_layer="fc"):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.fc_layer = fc_layer

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        conv_output = None
        for module_name, module in self.model._modules.items():

            if module_name == self.fc_layer:
                return conv_output, x

            try:
                x = module(x)
            except TypeError:
                x = nn.Sequential(*module)(x)  # Forward

            # print(module_name, module)
            # print(module_name)
            # print("x.requires_grad", x.requires_grad)
            if module_name == self.target_layer:
                # print('True')
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        if self.target_layer == self.fc_layer:
            return conv_output, x
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        rest_modules = split_by_module_name(self.model, self.fc_layer)
        for m, module in rest_modules.items():
            x = module(x)
        return conv_output, x


class GradCam:
    def __init__(self, model, target_layer, fc_layer="fc"):
        self.model = model
        self.model.eval()
        self.fc_layer = fc_layer
        self.extractor = CamExtractor(self.model, target_layer, fc_layer)

    def generate_cam(self, input_image, target_index=None, size=(224, 224)):

        if isinstance(size, int):
            size = (size, size)

        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_index is None:
            target_index = np.argmax(model_output.data.numpy())

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_index] = 1  # target_index -> 1, else -> 0
        one_hot_output = one_hot_output.to(model_output.device)

        self.model._modules[self.fc_layer].zero_grad()

        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        # Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2))
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = cv2.resize(cam, size)
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam


def get_class_activation_on_image(org_img, activation_map, size=(224, 224)):
    if isinstance(org_img, PIL.Image.Image):
        org_img = np.array(org_img)

    if isinstance(size, int):
        size = (size, size)

    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    org_img = cv2.resize(org_img, size)
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    img_with_heatmap = np.uint8(255 * img_with_heatmap)

    return img_with_heatmap


def get_img(npimg):
    return Image.fromarray(npimg)


def preprocess_image(pilimg, size=(224, 224)):

    if isinstance(size, int):
        size = (size, size)

    tran = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return tran(pilimg)


def split_by_module_name(model, module_name):
    modulelist = list(model._modules.keys())
    rest_module_names = modulelist[modulelist.index(module_name) :]
    return {m: model._modules[m] for m in rest_module_names}


if __name__ == "__main__":
    from aiscli.thirdparty.model.efficientnet_pretrained import *

    model = efficientnet_b0_pretrained(num_classes=500).cuda()
    image_path = "/home/ubuntu/yha/pytorch-cnn-visualizations/input_images/cat_dog.png"
    # image = cv2.imread(image_path)
    image = Image.open(image_path)
    image_prep = preprocess_image(image)
    image_prep = image_prep.unsqueeze(0).cuda()

    grad_cam = GradCam(model, target_layer="_conv_head", fc_layer="_fc")

    cam = grad_cam.generate_cam(image_prep, 1)
