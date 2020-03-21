import torch
import torch.nn.functional as F
import PIL
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
class GradCAM():
 def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module):
        self.model_arch = arch

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
 @classmethod
 def from_config(cls, arch: torch.nn.Module, model_type: str, layer_name: str):
    target_layer = layer_finders[model_type](arch, layer_name)
    return cls(arch, target_layer)

 def saliency_map_size(self, *input_size):
    device = next(self.model_arch.parameters()).device
    self.model_arch(torch.zeros(1, 3, *input_size, device=device))
    return self.activations['value'].shape[2:] , device

 def forward(self, input, class_idx=None, retain_graph=False):
    b, c, h, w = input.size()

    logit = self.model_arch(input)
    if class_idx is None:
        score = logit[:, logit.max(1)[-1]].squeeze()
    else:
        score = logit[:, class_idx].squeeze()

    self.model_arch.zero_grad()
    score.backward(retain_graph=retain_graph)
    gradients = self.gradients['value']
    activations = self.activations['value']
    b, k, u, v = gradients.size()

    alpha = gradients.view(b, k, -1).mean(2)
    # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
    weights = alpha.view(b, k, 1, 1)

    saliency_map = (weights*activations).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)
    saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
    self.gradients.clear()
    self.activations.clear()
    return saliency_map, logit

    
 def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
 

 def visualize_cam(mask, img, alpha=1.0):
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result
