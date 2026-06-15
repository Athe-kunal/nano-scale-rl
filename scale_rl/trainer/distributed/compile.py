# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from collections.abc import Callable

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
