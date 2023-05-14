from .expert import UpstreamExpert as _UpstreamExpert

def armhubert(ckpt, model_config, **kwargs):
    """
    FitHuBERT model

    Args:
        ckpt: 
            The checkpoint path for loading the model weights.
        model_config:
            The yaml config path for constructing the model.
    """
    return _UpstreamExpert(ckpt, model_config, **kwargs)
