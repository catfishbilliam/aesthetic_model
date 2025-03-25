import torch
from model import MyAestheticModel  # Your model implementation

def aesthetic_model(pretrained=True, **kwargs):
    # Delay the import inside the function
    from model import MyAestheticModel  # or from curate import MyAestheticModel if that's your file name
    model = MyAestheticModel(**kwargs)
    if pretrained:
        # Load pretrained weights (adjust URL/path as needed)
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://github.com/catfishbilliam/aesthetic_model/releases/download/v1.0/model.pth',
            progress=True
        )
        model.load_state_dict(checkpoint)
    model.eval()
    return model

