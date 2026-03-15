# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 Padding API"""

from typing import List, Optional

import torch

import transformer_engine_torch as tex

from ..fp8 import FP8GlobalStateManager
from ..jit import no_torch_dynamo
from ..tensor.float8_blockwise_tensor import Float8BlockwiseQTensor


__all__ = ["Fp8Unpadding"]


class _Fp8Unpadding(torch.autograd.Function):
    """functional FP8 unpadding"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        padded_m_splits: List[int],
        is_grad_enabled: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        inputmats = torch.split(inp.view(-1, inp.shape[-1]), padded_m_splits)
        out_ret = torch.cat(
            [mat[: m_splits[i]] for i, mat in enumerate(inputmats)],
            dim=0,
        )

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad = inp.requires_grad

        return out_ret

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring
        grad_input = None
        if ctx.requires_dgrad:
            grad_output = grad_output.contiguous()

            in_features = grad_output.shape[-1]

            fp8_blockwise_recipe = isinstance(grad_output, Float8BlockwiseQTensor)
            if fp8_blockwise_recipe:
                dtype = grad_output.dtype
                fp8_dtype = grad_output._fp8_dtype
                data_format = grad_output._data_format
                is_scaling_aware_transpose = grad_output._is_scaling_aware_transpose
                scale = grad_output._rowwise_scale_inv
                grad_output = grad_output._rowwise_data

            # Allocate cast and transpose output tensor
            total_row = sum(ctx.padded_m_splits)
            grad_input = torch.empty(
                [total_row, in_features], dtype=grad_output.dtype, device=grad_output.device
            )
            # FP8 pad input for forward, FP8 input transpose for backward wgrad
            tex.fused_multi_row_padding(
                grad_output.view(-1, in_features), grad_input, ctx.m_splits, ctx.padded_m_splits
            )

            if fp8_blockwise_recipe:
                scale_out = torch.empty(
                    [total_row, scale.shape[-1]], dtype=scale.dtype, device=scale.device,
                )
                tex.fused_multi_row_padding(
                    scale, scale_out, ctx.m_splits, ctx.padded_m_splits,
                )
                grad_input = Float8BlockwiseQTensor(
                    shape=grad_input.shape,
                    dtype=dtype,
                    rowwise_data=grad_input,
                    rowwise_scale_inv=scale_out,
                    columnwise_data=None,
                    columnwise_scale_inv=None,
                    fp8_dtype=fp8_dtype,
                    quantizer=None,
                    is_2D_scaled=False,
                    requires_grad=grad_input.requires_grad,
                    data_format=data_format,
                    is_scaling_aware_transpose=is_scaling_aware_transpose,
                )

        return (grad_input, None, None, None)


class Fp8Unpadding(torch.nn.Module):
    """
    Apply the unpadding for Grouped GEMM input.

    Parameters
    ----------
    num_gemms : int
                number of GEMMs to be performed simultaneously.
    align_size : int, optional
                 the alignment size for the input tensor. If not provided, the alignment size will
                 be determined by the FP8 recipe (32 for MXFP8 and 16 for others) in the first
                 forward pass.
    """

    def __init__(
        self,
        num_gemms: int,
        align_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.num_gemms = num_gemms
        self.align_size = align_size

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
    ) -> torch.Tensor:
        """
        Apply the unpadding to the input.

        Parameters
        ----------
        inp : torch.Tensor
                Input tensor.
        m_splits : List[int]
                    List of integers representing the split of the input tensor.
        """

        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."
        if self.align_size is None:
            self.align_size = 32 if FP8GlobalStateManager.get_fp8_recipe().mxfp8() else 16

        # FP8 padding calculate
        padded_m_splits = [
            (m + self.align_size - 1) // self.align_size * self.align_size for m in m_splits
        ]
        # no padding needed
        if m_splits == padded_m_splits:
            return inp

        if torch.is_grad_enabled():
            fn = _Fp8Unpadding.apply
            args = []
        else:
            fn = _Fp8Unpadding.forward
            args = [None]

        args += (
            inp,
            m_splits,
            padded_m_splits,
            torch.is_grad_enabled(),
        )
        out = fn(*args)

        return out
