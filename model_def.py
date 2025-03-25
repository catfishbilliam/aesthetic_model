class MyAestheticModel:
    def __init__(self, **kwargs):
        pass
    def load_state_dict(self, checkpoint):
        pass
    def eval(self):
        pass
    def to(self, device):
        return self
    def __call__(self, input_tensor):
        return 0.5  # Dummy score
