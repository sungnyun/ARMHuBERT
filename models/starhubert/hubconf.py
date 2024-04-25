from .expert import UpstreamExpert as _UpstreamExpert

def starhubert(ckpt, model_config, **kwargs):
    """
    FSPHuBERT model

    Args:
        ckpt: 
            The checkpoint path for loading the model weights.
        model_config:
            The yaml config path for constructing the model.
    """
    return _UpstreamExpert(ckpt, model_config, **kwargs)
