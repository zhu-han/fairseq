import torch
import random

from .attention import MultiHeadedAttention
from .embedding import PositionalEncoding
from .encoder_layer import EncoderLayer
from .layer_norm import LayerNorm
from .positionwise_feed_forward import PositionwiseFeedForward
from .repeat import repeat
from .subsampling import Conv2dSubsampling
from .embedding import PositionalEncoding

class Encoder(torch.nn.Module):
    """Transformer encoder module

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param int domain_dim: domain dim
    """

    def __init__(self, idim,
                 eprojs=1024,
                 eunits=1024,
                 elayers=6,
                 dropout_rate=0.1,
                 input_layer="conv2d",
                 pos_enc=None,
                 normalize_before=True,
                 concat_after=False,
                 ):
        super(Encoder, self).__init__()
        if pos_enc is None:
            pos_enc = PositionalEncoding(attention_dim, positional_dropout_rate)
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate, pos_enc)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim),
                pos_enc
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc,
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        self.encoders = torch.nn.LSTM(
                attention_dim,
                attention_dim//2,
                num_blocks,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=True,
            )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param torch.Tensor tags: input domain id
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """

        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)            
        xs, masks = self.encoders(xs, masks, None)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
        
    def feature_extractor(self, xs, masks):
        return self.embed.forward_nopos(xs, masks)

    def transformer_forward(self, xs, masks):
        xs = self.embed.forward_addpos(xs)
        xs, masks = self.encoders(xs, masks, None)
        return xs, masks