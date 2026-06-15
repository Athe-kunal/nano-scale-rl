# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

import torch
import torch.nn.functional as F
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
    sdpa_kernel,
    SDPBackend,
)
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _mask_mod_signature,
    _score_mod_signature,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.nn.attention.varlen import varlen_attn

