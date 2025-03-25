import torch
from model import MyAestheticModel  # Your model implementation

def aesthetic_model(pretrained=True, **kwargs):
    """
    Loads and returns the MyAestheticModel model.
    If pretrained=True, loads the pretrained weights.
    """
    model = MyAestheticModel(**kwargs)
    if pretrained:
        # You can download weights or load from a local file.
        # For example, if your repo has a weights file, you can load it here.
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://github.com/catfishbilliam/aesthetic_model/releases/download/v1.0/model.pth',
            progress=True
        )
        model.load_state_dict(checkpoint)
    model.eval()
    return model
