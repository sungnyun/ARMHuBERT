from dataclasses import dataclass, field

import torch
import torch.nn as nn
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel
from fairseq.modules import GradMultiply, LayerNorm
# from fairseq.data.data_utils import compute_mask_indices
import numpy as np

from .module import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
    ConformerEncoder,
    TransformerSentenceEncoderLayer,
    SplitLinear,
    LayerWiseProjHead
)

@dataclass
class CustomStudentModelConfig(FairseqDataclass):

    extractor_mode: str = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
            "Choose from ['default', 'layer_norm']"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: str = field(
        default="gelu",
        metadata={
            "help": "activation function to use"
            "Choose from ['relu', 'gelu', 'gelu_fast', 'gelu_accurate', 'tanh', 'linear']"
        }
    )
    layer_grad_scale: bool = field(
        default=False, metadata={"help": "apply layer gradient scaling"}
    )

    layer_type: str = field(
        default="transformer",
        metadata={
            "help": "layer type in encoder"
            "Choose from ['transformer', 'conformer']"
        }
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )

    # masking
    freeze_student_mask: bool = field(
        default=False, metadata={"help": "freeze student mask embedding paramter"}
    )
    update_teacher_mask: bool = field(
        default=False, metadata={"help": "update taecher mask emb to the student's at every step"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    pos_conv_depth: int = field(
        default=1,
        metadata={"help": "depth of positional encoder network"},
    )
    max_positions: int = field(default=100000, metadata={"help": "Max positions"})
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )
    crop_seq_to_multiple: int = field(
        default=1,
        metadata={
            "help": "crop convolutional feature extractor output such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})

    # Model Initialization
    init_conv_layers: bool = field(
        default=False,
        metadata={"help": "Whether initialize conv layer of teacher model or not"}
    )

    init_encoder_layers: int = field(
        default=0,
        metadata={"help": "# of layer to initialize encoder layer of teacher model."
                          "For non-positive integer, recognize as False"}
    )

    # Prediction Head
    pred_head_inter_dim: int = field(
        default=0,
        metadata={"help": "Intermediate dimension of prediction head"}
    )

    pred_head_final_dim: int = field(
        default=768,
        metadata={"help": "Final output dimension of prediction head"
                          "Same as transformer hidden dimension of teacherm model"}
    )

    pred_layer_id: str = field(
        default="[3, 7, 11]",
        metadata={"help": "Layer index to predict by prediction heads"}
    )

    layerwise_proj: bool = field(
        default=False,
        metadata={"help": "Whether to use (naive) layer-wise projection for distillation"}
    )

    # Time-reduction Layer
    enable_tr_layer: bool = field(
        default=True,
        metadata={"help": "applying time reduction layer or not"}
    )

    tr_reduce_factor: int = field(
        default=2,
        metadata={"help": "Factor to reduce time length"}
    )
    
    tr_layer_type: str = field(
        default="fc1",
        metadata={"help": "type of time reduction layer"
                          "fc1 or fc2 or conv1d"}
    )
    
    tr_conv1d_kernel: int = field(
        default= 2,
        metadata={"help": "If tr is conv1d, kernel for conv1d"
                          "stride is fixed to <tr_reduce_factor>"}
    )
    
    tr_layer_index: int = field(
        default=1,
        metadata={"help": "In which index should the time reduction layer be inserted"}
    )

    _teacher_task_agnostic: bool = field(
        default=False,
        metadata={"help": "Flag to determine whether the teacher model is task-agnostic"
                          "Do not assign value to this variable manually"}
    )

    _cnn_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of CNN_loss"
                          "Do not assign value to this variable manually"}
    )

class CustomStudentModel(BaseFairseqModel):
    def __init__(
        self, 
        cfg: CustomStudentModelConfig,
        teacher_model=None, # Need to be Wrapper class. Please check utils/utils.py
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg

        # Must be turned off for using cnn feature extractor
        # assert cfg.enable_log_mel == False
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0] # final embedding dimension of feature extractor
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
            ) 

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.feature_grad_mult = cfg.feature_grad_mult

        encoder_cls = TransformerEncoder
        if cfg.layer_type == "conformer" and cfg.pos_enc_type in ["rel_pos", "rope"]:
            encoder_cls = ConformerEncoder

        self.encoder = encoder_cls(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.init_conv_layers = cfg.init_conv_layers
        self.init_encoder_layers = cfg.init_encoder_layers
        self._teacher_task_agnostic = cfg._teacher_task_agnostic

        if self.init_conv_layers:
            assert teacher_model is not None
            self.init_from_teacher_conv(teacher_model)
        if self.init_encoder_layers > 0:
            assert teacher_model is not None
            self.init_from_teacher_enc(teacher_model, self.init_encoder_layers)

        inter_dim = cfg.pred_head_inter_dim
        pred_head_inter_dim = inter_dim if inter_dim > 0 else cfg.encoder_embed_dim
        pred_head_final_dim = cfg.pred_head_final_dim
        self.pred_layer_id = eval(cfg.pred_layer_id)
        self.n_tasks = len(self.pred_layer_id)

        self.enable_tr_layer = cfg.enable_tr_layer
        self.upsampler = None
        if cfg.enable_tr_layer:
            self.upsampler = torch.nn.ConvTranspose1d(
                in_channels=cfg.encoder_embed_dim,
                out_channels=cfg.encoder_embed_dim,
                kernel_size=cfg.tr_reduce_factor,
                stride=cfg.tr_reduce_factor,
            )

        self.layerwise_proj = cfg.layerwise_proj
        if cfg.layerwise_proj:
            # (Naive) Layer-wise projection
            # TODO: Try split version vs single version
            self.proj_head = nn.ModuleList([
                LayerWiseProjHead(
                    in_dim=cfg.encoder_embed_dim,
                    out_dim=cfg.pred_head_final_dim,
                    enable_tr_layer=cfg.enable_tr_layer,
                    tr_reduce_factor=cfg.tr_reduce_factor,
                ) for _ in range(cfg.encoder_layers)
            ])
        else:
            # DistilHuBERT style projection
            self.proj_head = nn.Sequential(
                nn.Linear(cfg.encoder_embed_dim, pred_head_inter_dim * self.n_tasks),
                nn.GELU(),
                SplitLinear(pred_head_inter_dim, self.n_tasks, pred_head_final_dim),
            ) if self.n_tasks > 0 else None
        
        self.final_proj = None
        self.specaug = None

    def apply_mask(self, x, padding_mask, mask_indices=None, mask_channel_indices=None):
        # B, T, C = x.shape
        if mask_indices is not None:
            assert x.shape[:2] == mask_indices.shape
            x[mask_indices] = self.mask_emb.to(x.dtype)
        if mask_channel_indices is not None:
            raise NotImplementedError
        return x

    def add_specaug(self, specaug):
        self.specaug = specaug

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def _disable_projection_heads(self):
        if self.layerwise_proj:
            self.final_proj = self.proj_head[-1]
            self.proj_head = None
        else:
            self.proj_head = None
        # self.cnn_proj_head = None

    def _upsample(self, x):
        if self.upsampler:
            x = x.transpose(1,2)
            x = self.upsampler(x)
            x = x.transpose(1,2)

        return x


    def forward(
        self,
        source,
        padding_mask=None,
        layer=None,
        mask_indices=None,
    ):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None and padding_mask.any():
            
            # feature: B X T' X D
            # padding_mask: B X T
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )
            # padding_mask: B X T'

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
        if time_steps_to_drop != 0:
            features = features[:, :-time_steps_to_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        # apply mask as the teacher's mask
        if mask_indices is not None:
            features = self.apply_mask(features, padding_mask, mask_indices)      

        x, layer_results, tr_layer_results = self.encoder(features, padding_mask=padding_mask, layer=layer)

        if self.layerwise_proj:
            if self.proj_head:
                projections = [
                    head(layer_results[i].transpose(0, 1))
                    for i, head in enumerate(self.proj_head)
                ]
                x = projections[-1]
            else:
                x = self.final_proj(x)
                projections = None

        else:
            if self.enable_tr_layer: # -> if not layerwise_proj
                x = self._upsample(x)

            if self.proj_head:
                # DistilHuBERT style projection
                b_sz, t_sz, _ = x.shape
                pred = self.proj_head(x)
                projections = (
                    pred
                    .reshape(b_sz, t_sz, self.n_tasks, -1)
                    .permute(0, 2, 1, 3)
                ) # B x N x T x D
            else:
                projections = None

        return {
            "x": x,
            "padding_mask": padding_mask,
            "layer_results": layer_results,
            "tr_layer_results": tr_layer_results,
            "projections": projections
        }

    def extract_features(self, source, padding_mask, layer=None):
        res = self.forward(
            source, padding_mask, layer=layer
        )
        return res

    def init_from_teacher_conv(self, teacher_model):
        if not self._teacher_task_agnostic:
            teacher_model = teacher_model.model.w2v_encoder.w2v_model

        self.feature_extractor.load_state_dict(
            teacher_model.model.feature_extractor.state_dict()
        )
        try:
            self.post_extract_proj.load_state_dict(
                teacher_model.model.post_extract_proj.state_dict()
            )
        except:
            pass


    def init_from_teacher_enc(self, teacher_model, n_layers):
        assert n_layers <= self.cfg.encoder_layers

        if not self._teacher_task_agnostic:
            teacher_model = teacher_model.model.w2v_encoder.w2v_model

        self.encoder.pos_conv.load_state_dict(
            teacher_model.model.encoder.pos_conv.state_dict()
        )

        for i in range(n_layers):
            self.encoder.layers[i].load_state_dict(
                teacher_model.model.encoder.layers[i].state_dict()
            )
