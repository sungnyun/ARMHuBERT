import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
    GradMultiply,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import index_put
from fairseq.modules.conformer_layer import ConformerWav2Vec2EncoderLayer
from fairseq.modules import RelPositionalEncoding

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


def pad_to_multiple(x, multiple, dim=-1, value=0):
        # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
        if x is None:
            return None, 0
        tsz = x.size(dim)
        m = tsz / multiple
        remainder = math.ceil(m) * multiple - tsz
        if m.is_integer():
            return x, 0
        pad_offset = (0,) * (-1 - dim) * 2

        return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


class TransformerEncoder(nn.Module):
    def build_encoder_layer(self, args):
        if args.layer_type == "transformer":
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
            )
        elif args.layer_type == "conformer":
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=args.encoder_ffn_embed_dim,
                attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                activation_fn="swish",
                attn_type=args.attn_type,
                use_fp16=args.fp16,
                pos_enc_type="abs",
            )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding=k // 2,
                                groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(l)
                    ]
                )

            self.pos_conv = make_conv_block(
                self.embedding_dim, k, args.conv_pos_groups, num_layers
            )

        else:
            def make_conv_pos(e, k, g):
                pos_conv = nn.Conv1d(
                    e,
                    e,
                    kernel_size=k,
                    padding=k // 2,
                    groups=g,
                )
                dropout = 0
                std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
                nn.init.normal_(pos_conv.weight, mean=0, std=std)
                nn.init.constant_(pos_conv.bias, 0)

                pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
                pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

                return pos_conv

            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups,
            )

        # Time-reduction layer
        if not args.enable_tr_layer:
            tr_layer = None
        else:
            self.tr_reduce_factor = args.tr_reduce_factor
            if args.tr_layer_type == 'fc1':
                # Input length will be verified first.
                tr_layer = nn.Linear(
                    self.embedding_dim * args.tr_reduce_factor,
                    self.embedding_dim
                )
                nn.init.xavier_uniform_(tr_layer.weight)

            elif args.tr_layer_type == 'fc2':
                tr_layer = nn.Sequential(
                    nn.Linear(self.embedding_dim * args.tr_reduce_factor, self.embedding_dim * args.tr_reduce_factor),
                    nn.GELU(),
                    nn.Linear(self.embedding_dim * args.tr_reduce_factor, self.embedding_dim),
                )
                
            elif args.tr_layer_type == 'conv1d':
                tr_layer = nn.Conv1d(
                    self.embedding_dim,
                    self.embedding_dim,
                    kernel_size=args.tr_reduce_factor,
                    stride=args.tr_reduce_factor
                )
            
            else:
                raise NotImplementedError(
                    "Wrong type of time reduction layer."
                    "Time reduction layers must be one of ['fc1', 'fc2', 'conv1d']."
                )

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.num_encoder_layers = len(self.layers)
        if args.enable_tr_layer:
            self.layers.insert(args.tr_layer_index, tr_layer)

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results, tr_layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        elif self.layer_norm_first and layer >= len(self.layers):
            x = self.layer_norm(x)

        return x, layer_results, tr_layer_results

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
    ):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        tr_layer_results = []
        r = None

        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                if isinstance(layer, nn.Linear):
                    x = self.concat_channelwise(x)
                    x = layer(x)
                    tr_layer_results.append(x)

                    # time-reduce padding mask
                    if padding_mask is not None:
                        sp = padding_mask.split(self.tr_reduce_factor, 1)
                        if padding_mask.shape[-1] % self.tr_reduce_factor != 0:
                            sp = sp[:-1]
                        padding_mask = torch.stack(sp).any(-1).transpose(0,1)
                elif isinstance(layer, nn.Conv1d):
                    x = x.permute(1, 2, 0).contiguous()
                    x = layer(x)
                    x = x.permute(2, 0, 1).contiguous()
                    tr_layer_results.append(x)

                    # time-reduce padding mask
                    if padding_mask is not None:
                        sp = padding_mask.split(self.tr_reduce_factor, 1)
                        if padding_mask.shape[-1] % self.tr_reduce_factor != 0:
                            sp = sp[:-1]
                        padding_mask = torch.stack(sp).any(-1).transpose(0,1)
                else:
                    x, (z, lr) = layer(
                        x, self_attn_padding_mask=padding_mask, need_weights=False
                    )
                    ####################### Gradient scale ####################### 
                    if i < self.num_encoder_layers and self.args.layer_grad_scale:  # i starts from 1
                        layer_grad_mult = (self.num_encoder_layers - i) / (self.num_encoder_layers - i + 1)
                        x = GradMultiply.apply(x, layer_grad_mult)
                    if i >= min_layer:
                        if i < self.num_encoder_layers and self.args.layer_grad_scale:
                            layer_results.append((GradMultiply.apply(x, 1 - layer_grad_mult), z, lr))
                        else:
                            layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo padding
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:-pad_length],
                    b[:-pad_length] if b is not None else b,
                    c[:-pad_length],
                )
            
            layer_results = [undo_pad(*u)[0] for u in layer_results]
        else:
            layer_results = [lr[0] for lr in layer_results]

        return x, layer_results, tr_layer_results

    def concat_channelwise(self, x):
        # x is shaped T x B x C
        time_length, batch, channel = x.size()
        how_many_pad = self.tr_reduce_factor - time_length % self.tr_reduce_factor 
        if how_many_pad != 0:
            # zero_pad = torch.zeros([how_many_pad, batch, channel]).cuda()
            zero_pad = torch.zeros([how_many_pad, batch, channel])
            x = torch.cat([x, zero_pad], dim = 0)
        time_length += how_many_pad

        result = torch.tensor([])
        
        j = 0
        while (j < self.tr_reduce_factor):
            # (T / factor) X B x (C * factor)
            tensor_to_concat = x[j::self.tr_reduce_factor,:,:]
            result = torch.cat([result, tensor_to_concat], dim = 2)
            j += 1
        # (T / factor) X B X (C * factor)
        return result

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


class ConformerEncoder(TransformerEncoder):
    def build_encoder_layer(self, args):
        layer = ConformerWav2Vec2EncoderLayer(
            embed_dim=self.embedding_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
            activation_fn="swish",
            attn_type=args.attn_type,
            pos_enc_type=args.pos_enc_type,
            use_fp16=args.fp16,  # only used for rope
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.pos_enc_type = args.pos_enc_type
        max_source_positions = self.max_positions()

        if self.pos_enc_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                max_source_positions, self.embedding_dim
            )
        elif self.pos_enc_type == "rope":
            self.embed_positions = None
        else:
            raise Exception("Unsupported positional encoding type")

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # B X T X C here
        position_emb = None
        if self.pos_enc_type == "rel_pos":
            position_emb = self.embed_positions(x)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=self.need_weights,
                    position_emb=position_emb,
                )
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ):

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )
        self.self_attn._set_skip_embed_dim_check()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)


class SplitLinear(nn.Module):
    """Split Linear Layer"""

    def __init__(self, in_dim, in_split, out_dim):
        super().__init__()

        self.in_dim = in_dim  # Din
        self.in_split = in_split  # N
        self.out_dim = out_dim  # Dout

        if in_split > 1:
            weight = torch.zeros((self.in_split, self.in_dim, self.out_dim))
            self.weight = nn.Parameter(weight, requires_grad=True)
            nn.init.uniform_(self.weight, -(self.in_dim ** -0.5), self.in_dim ** -0.5)

            bias = torch.zeros((1, 1, self.in_split, self.out_dim))
            self.bias = nn.Parameter(bias, requires_grad=True)
            nn.init.uniform_(self.bias, -(self.in_dim ** -0.5), self.in_dim ** -0.5)
        else:
            self.layer = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x:torch.Tensor):
        # x: shape = B x T x NDin

        if self.in_split == 1:
            return self.layer(x)
        else:
            x = x.reshape(x.shape[0], x.shape[1], self.in_split, 1, self.in_dim)
            # x: B x T x N x 1 x Din

            out = torch.einsum("...klm,kmn->...kln", x, self.weight).squeeze(3)
            # out: B x T x N x Dout
            out = out + self.bias

            return out.reshape(x.shape[0], x.shape[1], -1) # -> B x T x NDout ?


class LayerWiseProjHead(nn.Module):
    """Projection Head for (naive) layer-wise distillation"""

    def __init__(self, in_dim, out_dim, enable_tr_layer, tr_reduce_factor):
        super().__init__()

        self.in_dim = in_dim  # Din
        self.out_dim = out_dim  # Dout
        self.enable_tr_layer = enable_tr_layer
        self.tr_reduce_factor = tr_reduce_factor

        self.upsampler = None
        if self.enable_tr_layer:
            self.upsampler = nn.ConvTranspose1d(
                in_channels=self.in_dim,
                out_channels=self.in_dim,
                kernel_size=self.tr_reduce_factor,
                stride=self.tr_reduce_factor,
            )
        
        self.lin_proj = None
        if self.in_dim != self.out_dim:
            self.lin_proj = nn.Linear(
                in_features=self.in_dim,
                out_features=self.out_dim,
            )

    def forward(self, x:torch.Tensor):
        # x: (B x T/f x D_in)

        if self.upsampler:
            x = x.transpose(1,2)
            x = self.upsampler(x)
            x = x.transpose(1,2)

        if self.lin_proj:
            x = self.lin_proj(x)

        # x: (B x T x D_out)
        return x
