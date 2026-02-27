# ruff: noqa
"""
Backwards compatibility shim for GPTQModifier.

This module has been moved to llmcompressor.modifiers.gptq.
This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "Importing from 'llmcompressor.modifiers.quantization.gptq' is deprecated. "
    "Please update your imports to use 'llmcompressor.modifiers.gptq' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

from llmcompressor.modifiers.gptq import *
from llmcompressor.modifiers.gptq.base import GPTQModifier

__all__ = ["GPTQModifier"]
