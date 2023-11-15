# Copyright 2024 the authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
import netket as nk
import numpy.typing as npt

from .utils import REAL_DTYPE, circulant


class MultiLayerPerceptron(nn.Module):
    """Flax module for a Multi-layer perceptron architecture with normalization.

    Args:
        layer_widths: Sequence of integers that define both the number of layers
            and their widths.
        activation_funcion: The activation function that will be applied after
            each dense layer.
        kernel_init: Function to initialize the trainable parameters.

    Returns:
        A jax array with the output of the net.
    """

    layer_widths: Sequence[int]
    activation_function: Callable = nn.swish
    kernel_init: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x) -> jt.ArrayLike:
        for w in self.layer_widths:
            # We cannot use LayerNorm when the output has size 1, since
            # that would destroy the data.
            if w == 1:
                normalizer = lambda x: x
            else:
                normalizer = nn.LayerNorm(param_dtype=REAL_DTYPE)
            x = self.activation_function(
                normalizer(
                    nn.Dense(
                        w,
                        kernel_init=self.kernel_init,
                        param_dtype=REAL_DTYPE,
                    )(x)
                )
            )
        return x


class AffinityPosWeight(nn.Module):
    "Flax module that multiplies by a circulant matrix."

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        weight_row = self.param(
            "alpha_delta",
            nn.initializers.truncated_normal(
                stddev=jnp.sqrt(1.0 / x.shape[-2])
            ),
            (x.shape[-2],),
            REAL_DTYPE,
        )

        weight = circulant(weight_row)

        return weight @ x


class PositionalHead(nn.Module):
    """Flax module that implements to a single head of linearized attention.

    Args:
        head_size: The dimension of each of the heads.
    """

    head_size: int

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        value = nn.Dense(
            self.head_size, use_bias=False, param_dtype=REAL_DTYPE
        )
        aff = AffinityPosWeight()
        return aff(value(x))


class MultiHeadPositionalAttention(nn.Module):
    """Flax module implementing a multi-head block.

    The input is split into the chosen number of heads and the result is
    concatenated.

    Args:
        n_heads: The number of heads present in the multi-head attention block.
        head_size: The dimension of each one of the heads.
    """

    n_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        heads = [PositionalHead(self.head_size) for _ in range(self.n_heads)]
        return jnp.concatenate([h(x) for h in heads], axis=-1)


class CoreBlock(nn.Module):
    """Flax module implementing the core block of the ViT.

    This comprises the embedding starting from the spin representation, the
    application of the multi-head linearized attention block and the multi-layer
    perceptron, and a final pass through a log-cosh function to achieve higher
    accuracy in highly ordered phases. All the operations here are applied to
    all tokens indistincly.

    Args:
        n_heads: The number of heads present in the multi-head attention block.
        n_ffn_layers: The number of layers present in the multi-layer perceptron.
    """

    n_heads: int
    n_ffn_layers: int

    @nn.compact
    def __call__(self, x) -> jt.ArrayLike:
        embedding_d = x.shape[-1]
        if embedding_d % self.n_heads != 0:
            raise ValueError(
                "The number of heads must divide the embedding dimensions"
            )
        head_size = embedding_d // self.n_heads
        sa = MultiHeadPositionalAttention(self.n_heads, head_size)
        x += sa(nn.LayerNorm(param_dtype=REAL_DTYPE)(x))
        ffn = MultiLayerPerceptron(
            [
                embedding_d,
            ]
            * self.n_ffn_layers
        )
        # No LayerNorm here because it is already included in the perceptron.
        return nk.nn.log_cosh(ffn(x) + x)


class RealSpinViT(nn.Module):
    """Flax module that implements a real-valued ViT architecture.

    Two of these modules can be combined to create a complex output. The input
    is expected to be passed configuration by configuration. Note that the
    tokenization is not including.

    Args:
        embedding_d: The dimension of the linear transformation that maps the
            binary spin representation into the feature space onto which the ViT
            will act.
        n_heads: The number of heads present in the Multi-Head attention block.
        n_blocks: The number of core blocks present in the model.
        n_ffn_layers: The number of layers present in the multi-layer perceptron
            contained in the CoreBlock.
        final_architecture: A sequence of integer widths for the layers in the
            post-processing MLP.
    """

    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]

    @nn.compact
    def __call__(self, x):
        embedding = nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)

        x = embedding(x)

        blocks = [
            CoreBlock(self.n_heads, self.n_ffn_layers)
            for _ in range(self.n_blocks)
        ]

        for cb in blocks:
            x = cb(x)

        # Sum over tokens (set pooling operation).
        x = x.sum(axis=0)

        postprocessor = MultiLayerPerceptron(self.final_architecture)
        x = postprocessor(x)

        # Fix the offset and scale.
        return nn.Dense(1, param_dtype=REAL_DTYPE)(x).squeeze()


class SpinViTWorker(nn.Module):
    """Flax module that implements a real- or complex-valued ViT architecture.

    Args:
        token_size: The number of contiguous spins from the chain to be grouped
            into each token.
        embedding_d: The dimension of the linear transformation that maps the
            binary spin representation into the feature space onto which the ViT
            will act.
        n_heads: The number of heads present in the Multi-Head attention block.
        n_blocks: The number of core blocks present in the model.
        n_ffn_layers: The number of layers present in the multi-layer perceptron
            contained in the CoreBlock.
        final_architecture: A sequence of integer widths for the layers in the
            post-processing MLP.
        is_complex: If True, two ViT architectures are defined, one for the real
            and one for the imaginary part of the wavefunction. Otherwise, no
            imaginary part is computed.
    """

    token_size: int
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]
    is_complex: bool

    @nn.compact
    def __call__(self, x):
        real_part = RealSpinViT(
            self.embedding_d,
            self.n_heads,
            self.n_blocks,
            self.n_ffn_layers,
            self.final_architecture,
        )(x)

        if self.is_complex:
            imag_part = RealSpinViT(
                self.embedding_d,
                self.n_heads,
                self.n_blocks,
                self.n_ffn_layers,
                self.final_architecture,
            )(x)

            return real_part + 1.0j * imag_part

        return real_part


class SpinViT(nn.Module):
    """Flax module wrapping `SpinViTWorker` and enforcing translation invariance.

    This is achieved by averaging the result of `SpinViTWorker` over all
    possible cyclic permutations.

    See the documentation of `SpinViTWorker` for information about the
    parameters.
    """

    token_size: int
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]
    is_complex: bool

    @nn.compact
    def __call__(self, x):
        worker = SpinViTWorker(
            self.token_size,
            self.embedding_d,
            self.n_heads,
            self.n_blocks,
            self.n_ffn_layers,
            self.final_architecture,
            self.is_complex,
        )

        circulant_x = circulant(x, self.token_size).reshape(
            (self.token_size, -1, self.token_size)
        )
        return jax.vmap(worker, in_axes=0)(circulant_x).mean(axis=0)


class BatchedSpinViT(nn.Module):
    "Batched version of SpinViT, accepting several spin configurations at once."

    token_size: int
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]
    is_complex: bool

    @nn.compact
    def __call__(self, batched_x):
        worker = SpinViT(
            self.token_size,
            self.embedding_d,
            self.n_heads,
            self.n_blocks,
            self.n_ffn_layers,
            self.final_architecture,
            self.is_complex,
        )
        return jax.vmap(worker, in_axes=0)(batched_x)
