import warnings
import copy
import math
import inspect
import collections
from collections import OrderedDict
import functools
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, TypeVar, Mapping
from itertools import repeat
from types import FunctionType
from enum import Enum

import torch
from torch.hub import load_state_dict_from_url
from torch import Tensor
import torch.nn as nn

from _meta import _IMAGENET_CATEGORIES
import functional as F

# from . import functional as F, InterpolationMode
class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``nearest-exact``, ``bilinear``, ``bicubic``, ``box``, ``hamming``,
    and ``lanczos``.
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"

# from ..ops.misc import Conv2dNormActivation, MLP
class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )

class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        _log_api_usage_once(self)


# from ..transforms._presets import ImageClassification, InterpolationMode
class ImageClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )


# from ..utils import _log_api_usage_once, _make_ntuple
def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


# from ._api import register_model, Weights, WeightsEnum
M = TypeVar("M", bound=nn.Module)

BUILTIN_MODELS = {}
def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn

    return wrapper

class Weights:
    """
    This class is used to group important attributes associated with the pre-trained weights.

    Args:
        url (str): The location where we find the weights.
        transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
            needed to use the model. The reason we attach a constructor method rather than an already constructed
            object is because the specific object might have memory and thus we want to delay initialization until
            needed.
        meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
            informative attributes (for example the number of parameters/flops, recipe link/methods used in training
            etc), configuration parameters (for example the `num_classes`) needed to construct the model or important
            meta-data (for example the `classes` of a classification model) needed to use the model.
    """
    url: str
    transforms: Callable
    meta: Dict[str, Any]

    def __init__(self, url, transforms, meta):
        self.url = url
        self.transforms = transforms
        self.meta = meta
        
    def __eq__(self, other: Any) -> bool:
        # We need this custom implementation for correct deep-copy and deserialization behavior.
        # TL;DR: After the definition of an enum, creating a new instance, i.e. by deep-copying or deserializing it,
        # involves an equality check against the defined members. Unfortunately, the `transforms` attribute is often
        # defined with `functools.partial` and `fn = partial(...); assert deepcopy(fn) != fn`. Without custom handling
        # for it, the check against the defined members would fail and effectively prevent the weights from being
        # deep-copied or deserialized.
        # See https://github.com/pytorch/vision/pull/7107 for details.
        if not isinstance(other, Weights):
            return NotImplemented

        if self.url != other.url:
            return False

        if self.meta != other.meta:
            return False

        if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
            return (
                self.transforms.func == other.transforms.func
                and self.transforms.args == other.transforms.args
                and self.transforms.keywords == other.transforms.keywords
            )
        else:
            return self.transforms == other.transforms


class WeightsEnum(Enum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
    `Weights`.

    Args:
        value (Weights): The data class entry with the weight information.
    """

    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls[obj.replace(cls.__name__ + ".", "")]
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj

    def get_state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return load_state_dict_from_url(self.url, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"

    @property
    def url(self):
        return self.value.url

    @property
    def transforms(self):
        return self.value.transforms

    @property
    def meta(self):
        return self.value.meta

# from ._utils import _ovewrite_named_param, handle_legacy_interface
def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if not seq:
        return ""
    if len(seq) == 1:
        return f"'{seq[0]}'"

    head = "'" + "', '".join([str(item) for item in seq[:-1]]) + "'"
    tail = f"{'' if separate_last and len(seq) == 2 else ','} {separate_last}'{seq[-1]}'"

    return head + tail

D = TypeVar("D")
W = TypeVar("W", bound=WeightsEnum)
V = TypeVar("V")

def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def handle_legacy_interface(**weights: Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]):
    """Decorates a model builder with the new interface to make it compatible with the old.

    In particular this handles two things:

    1. Allows positional parameters again, but emits a deprecation warning in case they are used. See
        :func:`torchvision.prototype.utils._internal.kwonly_to_pos_or_kw` for details.
    2. Handles the default value change from ``pretrained=False`` to ``weights=None`` and ``pretrained=True`` to
        ``weights=Weights`` and emits a deprecation warning with instructions for the new interface.

    Args:
        **weights (Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]): Deprecated parameter
            name and default value for the legacy ``pretrained=True``. The default value can be a callable in which
            case it will be called with a dictionary of the keyword arguments. The only key that is guaranteed to be in
            the dictionary is the deprecated parameter name passed as first element in the tuple. All other parameters
            should be accessed with :meth:`~dict.get`.
    """

    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @kwonly_to_pos_or_kw
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, **kwargs: Any) -> M:
            for weights_param, (pretrained_param, default) in weights.items():  # type: ignore[union-attr]
                # If neither the weights nor the pretrained parameter as passed, or the weights argument already use
                # the new style arguments, there is nothing to do. Note that we cannot use `None` as sentinel for the
                # weight argument, since it is a valid value.
                sentinel = object()
                weights_arg = kwargs.get(weights_param, sentinel)
                if (
                    (weights_param not in kwargs and pretrained_param not in kwargs)
                    or isinstance(weights_arg, WeightsEnum)
                    or (isinstance(weights_arg, str) and weights_arg != "legacy")
                    or weights_arg is None
                ):
                    continue

                # If the pretrained parameter was passed as positional argument, it is now mapped to
                # `kwargs[weights_param]`. This happens because the @kwonly_to_pos_or_kw decorator uses the current
                # signature to infer the names of positionally passed arguments and thus has no knowledge that there
                # used to be a pretrained parameter.
                pretrained_positional = weights_arg is not sentinel
                if pretrained_positional:
                    # We put the pretrained argument under its legacy name in the keyword argument dictionary to have
                    # unified access to the value if the default value is a callable.
                    kwargs[pretrained_param] = pretrained_arg = kwargs.pop(weights_param)
                else:
                    pretrained_arg = kwargs[pretrained_param]

                if pretrained_arg:
                    default_weights_arg = default(kwargs) if callable(default) else default
                    if not isinstance(default_weights_arg, WeightsEnum):
                        raise ValueError(f"No weights available for model {builder.__name__}")
                else:
                    default_weights_arg = None

                if not pretrained_positional:
                    warnings.warn(
                        f"The parameter '{pretrained_param}' is deprecated since 0.13 and may be removed in the future, "
                        f"please use '{weights_param}' instead."
                    )

                msg = (
                    f"Arguments other than a weight enum or `None` for '{weights_param}' are deprecated since 0.13 and "
                    f"may be removed in the future. "
                    f"The current behavior is equivalent to passing `{weights_param}={default_weights_arg}`."
                )
                if pretrained_arg:
                    msg = (
                        f"{msg} You can also use `{weights_param}={type(default_weights_arg).__name__}.DEFAULT` "
                        f"to get the most up-to-date weights."
                    )
                warnings.warn(msg)

                del kwargs[pretrained_param]
                kwargs[weights_param] = default_weights_arg

            return builder(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper

def kwonly_to_pos_or_kw(fn: Callable[..., D]) -> Callable[..., D]:
    """Decorates a function that uses keyword only parameters to also allow them being passed as positionals.

    For example, consider the use case of changing the signature of ``old_fn`` into the one from ``new_fn``:

    .. code::

        def old_fn(foo, bar, baz=None):
            ...

        def new_fn(foo, *, bar, baz=None):
            ...

    Calling ``old_fn("foo", "bar, "baz")`` was valid, but the same call is no longer valid with ``new_fn``. To keep BC
    and at the same time warn the user of the deprecation, this decorator can be used:

    .. code::

        @kwonly_to_pos_or_kw
        def new_fn(foo, *, bar, baz=None):
            ...

        new_fn("foo", "bar, "baz")
    """
    params = inspect.signature(fn).parameters

    try:
        keyword_only_start_idx = next(
            idx for idx, param in enumerate(params.values()) if param.kind == param.KEYWORD_ONLY
        )
    except StopIteration:
        raise TypeError(f"Found no keyword-only parameter on function '{fn.__name__}'") from None

    keyword_only_params = tuple(inspect.signature(fn).parameters)[keyword_only_start_idx:]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> D:
        args, keyword_only_args = args[:keyword_only_start_idx], args[keyword_only_start_idx:]
        if keyword_only_args:
            keyword_only_kwargs = dict(zip(keyword_only_params, keyword_only_args))
            warnings.warn(
                f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional "
                f"parameter(s) is deprecated since 0.13 and may be removed in the future. Please use keyword parameter(s) "
                f"instead."
            )
            kwargs.update(keyword_only_kwargs)

        return fn(*args, **kwargs)

    return wrapper

__all__ = [
    "VisionTransformer",
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        self.self_attention_weights = None

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, self_attention_weights = self.self_attention(x, x, x, need_weights=True)
        self_attention_weights *= 1000
        self.self_attention_weights = self_attention_weights
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length       

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x) # (n, n_h * n_w, hidden_dim)
        n = x.shape[0] # n: batch_size
        
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x) # (n, n_h * n_w + 1, hidden_dim)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        
        # x = self.heads(x)
        


        return x


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META: Dict[str, Any] = {
    "categories": _IMAGENET_CATEGORIES,
}

_COMMON_SWAG_META = {
    **_COMMON_META,
    "recipe": "https://github.com/facebookresearch/SWAG",
    "license": "https://github.com/facebookresearch/SWAG/blob/main/LICENSE",
}


class ViT_B_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 86567656,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.072,
                    "acc@5": 95.318,
                }
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
        transforms=partial(
            ImageClassification,
            crop_size=384,
            resize_size=384,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 86859496,
            "min_size": (384, 384),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.304,
                    "acc@5": 97.650,
                }
            },
            "_ops": 55.484,
            "_file_size": 331.398,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 86567656,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 96.180,
                }
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_B_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88224232,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.912,
                    "acc@5": 92.466,
                }
            },
            "_ops": 4.409,
            "_file_size": 336.604,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=242),
        meta={
            **_COMMON_META,
            "num_params": 304326632,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.662,
                    "acc@5": 94.638,
                }
            },
            "_ops": 61.555,
            "_file_size": 1161.023,
            "_docs": """
                These weights were trained from scratch by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
        transforms=partial(
            ImageClassification,
            crop_size=512,
            resize_size=512,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 305174504,
            "min_size": (512, 512),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.064,
                    "acc@5": 98.512,
                }
            },
            "_ops": 361.986,
            "_file_size": 1164.258,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 304326632,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.146,
                    "acc@5": 97.422,
                }
            },
            "_ops": 61.555,
            "_file_size": 1161.023,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_32-c7638314.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 306535400,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.972,
                    "acc@5": 93.07,
                }
            },
            "_ops": 15.378,
            "_file_size": 1169.449,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_H_14_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
        transforms=partial(
            ImageClassification,
            crop_size=518,
            resize_size=518,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 633470440,
            "min_size": (518, 518),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.552,
                    "acc@5": 98.694,
                }
            },
            "_ops": 1016.717,
            "_file_size": 2416.643,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 632045800,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.708,
                    "acc@5": 97.730,
                }
            },
            "_ops": 167.295,
            "_file_size": 2411.209,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_SWAG_E2E_V1


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_16_Weights.IMAGENET1K_V1))
def vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    weights = ViT_B_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_32_Weights.IMAGENET1K_V1))
def vit_b_32(*, weights: Optional[ViT_B_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_32_Weights
        :members:
    """
    weights = ViT_B_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_L_16_Weights.IMAGENET1K_V1))
def vit_l_16(*, weights: Optional[ViT_L_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_16_Weights
        :members:
    """
    weights = ViT_L_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_L_32_Weights.IMAGENET1K_V1))
def vit_l_32(*, weights: Optional[ViT_L_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_32_Weights
        :members:
    """
    weights = ViT_L_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", None))
def vit_h_14(*, weights: Optional[ViT_H_14_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_h_14 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_H_14_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_H_14_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_H_14_Weights
        :members:
    """
    weights = ViT_H_14_Weights.verify(weights)

    return _vision_transformer(
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state
