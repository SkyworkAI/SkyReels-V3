# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .wan_multitalk_14B import multitalk_14B

WAN_CONFIGS = {
    "multitalk-14B": multitalk_14B,
}
