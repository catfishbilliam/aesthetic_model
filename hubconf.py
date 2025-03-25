import torch
from model_def import MyAestheticModel

def aesthetic_model(pretrained=True, **kwargs):
    model = MyAestheticModel(**kwargs)
    if pretrained:
        checkpoint = torch.load("model.pth", map_location="cpu")
        model.load_state_dict(checkpoint)
    model.eval()
    return model
