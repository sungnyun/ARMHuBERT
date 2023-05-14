import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pytz import timezone
from typing import Any, Dict, List, Optional

from fairseq import models, tasks, quantization_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, merge_with_parent
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from omegaconf.omegaconf import open_dict

from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from teacher_utils.hubert import HubertModel, HubertConfig
from teacher_utils.wavlm import WavLM, WavLMConfig

from pytorch_lightning.utilities.distributed import rank_zero_only


class TeacherWrapper(nn.Module):
    def __init__(
        self,
        model,
    ):
        """
        Wrapper for the teacher model 
        The wrapper makes it possible to get every intermediate outputs via hooks
        """
        super().__init__()
        self.model = model
        self._hook_layer_hiddens = []
        self._hook_post_cnn = []

        def generate_hook_handler(hiddens: List):
            def hook_handler(self, input, output):
                hiddens.append(output)

            return hook_handler

        self.model.post_extract_proj.register_forward_hook(
                generate_hook_handler(self._hook_post_cnn)
            )

        for module in self.model.encoder.layers:
            module.register_forward_hook(
                generate_hook_handler(self._hook_layer_hiddens) # -> but is it absolutely needed?
            )

    def extract_features(self, source, padding_mask, mask=False):
        self._hook_layer_hiddens.clear()
        result = {}

        _, _, mask_id = self.model.extract_features(
                        source,
                        padding_mask,
                        mask=mask,
        )

        hook_layer_hiddens = self._hook_layer_hiddens.copy()
        self._hook_layer_hiddens.clear()
        hook_post_cnn = self._hook_post_cnn.copy()
        self._hook_post_cnn.clear()

        result['layer_results'] = hook_layer_hiddens
        result['x'] = result['layer_results'][-1][0].transpose(0, 1)
        result['features'] = hook_post_cnn
        result['mask_indices'] = mask_id

        return result

def load_wavlm_and_config(filename, arg_overrides: Optional[Dict[str, Any]] = None):
    state = torch.load(filename)
    model_cfg = WavLMConfig(state["cfg"])
    ### overwrite cfg ###
    for k, v in arg_overrides.items():
        if k in model_cfg.__dir__():
            setattr(model_cfg, k, v)
        else:
            print(f'No attribute {k} in WavLMConfig')
            pass
    model = WavLM(model_cfg)
    model.load_state_dict(state["model"], strict=True)

    model.feature_grad_mult = 0.0
    model.encoder.layerdrop = 0.0

    # Wrap Teacher
    model = TeacherWrapper(model)
    task_agnostic = True

    return model, model_cfg, task_agnostic

def load_model_and_config(filename, arg_overrides: Optional[Dict[str, Any]] = None):

    state = load_checkpoint_to_cpu(filename, arg_overrides)

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    
    model_cfg = cfg.model
    model_type = getattr(model_cfg, "_name", None) # or getattr(cfg, "arch", None)
    task_agnostic = None

    if model_type == 'wav2vec2':
        model_cfg = merge_with_parent(Wav2Vec2Config(), model_cfg)
        with open_dict(model_cfg):
            model_cfg.required_seq_len_multiple = 1
        model = Wav2Vec2Model.build_model(model_cfg)
        task_agnostic = True
    elif model_type == "hubert":
        task = tasks.setup_task(cfg.task)
        task.load_state_dict(state["task_state"])
        model_cfg = merge_with_parent(HubertConfig(), model_cfg)
        # Update needed due to a bug in latest version of fairseq
        with open_dict(model_cfg):
            model_cfg.required_seq_len_multiple = 1
            model_cfg.layer_type = 'transformer'
        model = HubertModel.build_model(model_cfg, task)
        task_agnostic = True
    else:
        raise NotImplementedError(f"model '{model_type}' is not supported.")

    model = quantization_utils.quantize_model_scalar(model, cfg)
    model.load_state_dict(state['model'], strict=True, model_cfg=cfg.model)

    # Wrap Teacher
    model.encoder.layerdrop = 0
    model = TeacherWrapper(model)

    return model, model_cfg, task_agnostic


# Make yaml file with given name and config dataclass
@rank_zero_only
def dump_yaml(cfg, yaml_dict, time_tag):
    
    # cfg: updated distiller config dataclass (= student_config)
    # yaml_file: dumping yaml file (= YAML_CFG)
    distiller = dict()
    
    for attr in dir(cfg):
        if not callable(getattr(cfg, attr)) and not attr.startswith("_"):
            distiller[attr] = getattr(cfg, attr)

    dump_dict = yaml_dict

    for key in distiller:
        if key in ['activation_fn', 'extractor_mode', 'layer_type']:
            dump_dict['distiller'][key] = str(distiller[key])
        else:
            dump_dict['distiller'][key] = distiller[key]

    dump_dir = dump_dict['train']['base_dir'] + 'results/pretrain/' + dump_dict['train']['output_dir']
    os.makedirs(dump_dir, exist_ok=True)

    with open(os.path.join(dump_dir, time_tag + '.yaml'), 'w') as f:
        yaml.dump(dump_dict, f, sort_keys = False)
    
    return dump_dict


def get_time_tag():
    return datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%Hh%Mm%Ss')


def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False
