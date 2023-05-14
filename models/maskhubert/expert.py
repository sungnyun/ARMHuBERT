import yaml
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence

from .model import CustomStudentModelConfig, CustomStudentModel

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt, model_config, **kwargs):
        """
        Args:
            ckpt:
                The (lightning) checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()

        # Load model config
        with open(model_config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        model_config = cfg['distiller']

        # Disable initialization from teacher
        model_config['init_conv_layers'] = False
        model_config['init_encoder_layers'] = 0

        self.model_config = CustomStudentModelConfig(**model_config)

        # The model needs to be a nn.Module for finetuning, not required for representation extraction
        self.model = CustomStudentModel(self.model_config)
        
        # Load ckpt file for model weights
        def load_model_weights():
            state = torch.load(ckpt, map_location="cpu")
            state_dict = OrderedDict({k[14:]: v for k, v in state['state_dict'].items() if 'student_model' in k})
            self.model.load_state_dict(state_dict)

        load_model_weights()
        
        # self.model._disable_projection_heads()

    def get_downsample_rates(self, key: str):
        return 320

    def forward(self, wavs):
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        wav_lens = torch.LongTensor([len(wav) for wav in wavs])
        src = pad_sequence(wavs, batch_first=True)

        padding_mask = ~torch.lt(
            torch.arange(max(wav_lens)).unsqueeze(0),
            wav_lens.unsqueeze(1),
        )

        results = self.model(
            source=src,
            padding_mask=padding_mask,
        )

        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        # hidden_states = [lr.transpose(0, 1) for lr in results['layer_results']]
        return {
            "last_hidden_state": results['x'],
            "hidden_states": results['projections'],
        }
