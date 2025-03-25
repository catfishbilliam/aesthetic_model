import torch


def aesthetic_model(pretrained=True, **kwargs):
    from my_model import MyAestheticModel  # now importing from a different file
    model = MyAestheticModel(**kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://github.com/catfishbilliam/aesthetic_model/releases/download/v1.0/model.pth',
            progress=True
        )
        model.load_state_dict(checkpoint)
    model.eval()
    return model
