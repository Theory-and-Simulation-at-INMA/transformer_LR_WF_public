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

from typing import Optional, Tuple

import netket as nk
import numpy.typing as npt
from netket.operator.spin import sigmax, sigmaz


def get_Hamiltonian(
    N: int,
    J: float,
    alpha: float,
    trans_field: float = -1.0,
    sym_field: Optional[bool] = False,
    epsilon: Optional[float] = 1e-3,
    return_norm: Optional[float] = False,
) -> Tuple[npt.ArrayLike, float]:
    """Build the Hamiltonian of the system.

    Args:
        N: The number of spins in the chain.
        J: The bare spin-spin interaction strength.
        alpha: The exponent of the power-law governing the decay of the
            interaction strength.
        trans_field: The transverse component of an external field.
        sym_field: The longitudinal component of an external field.
            It lifts the degeneracy in the ordered phase breaking the symmetry.
        epsilon: The relative strength of the longitudinal component respect
            to the interaction strength.
        return_norm: Flag that allows the function to return the value of the
            normalization constant.

    Returns:
        Either H or the tuple (H, N_norm) depending on whether return_norm
        is True.
            H: Array containing the Hamiltonian matrix.
            N_norm: Float value of the normalization constant.
    """

    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    H = sum(trans_field * sigmax(hi, i) for i in range(N))

    N_norm = 1
    for i in range(1, N):
        dist = min(abs(i), N - abs(i))
        N_norm += 1 / dist**alpha

    J = J / N_norm

    for i in range(0, N):
        for j in range(i, N):
            dist = min(abs(i - j), N - abs(i - j))
            cn = 1.0
            if dist == 0:
                dist = 1
                cn = 2.0
            H += J / cn * sigmaz(hi, i) * sigmaz(hi, j) / (dist**alpha)

    if sym_field:
        H += J * epsilon * sum(sigmaz(hi, i) for i in range(N))

    H /= N
    if return_norm:
        return (H, N_norm)
    else:
        return H


def get_eigvals(
    Hamiltonian: npt.ArrayLike, order: int = 1, eigenvecs: bool = False
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Partially diagonalize a given matrix, namely the Hamiltonian.

    Args:
        Hamiltonian: Hamiltonian or general matrix to diagonalize.
        order: The number of lowest energy levels that we want to obtain.
        eigenvecs: If set to True, returns also an Array with the eigenvectors
            of the corresponding eigenlevels obtained.

    Returns:
        Either w or the tuple (w, v) depending on whether compute_eigenvectors
        is True.
            w: Array containing the lowest 'order' eigenvalues.
            v: Array containing the eigenvectors as columns, such that
                'v[:, i]' corresponds to w[i].
    """

    return nk.exact.lanczos_ed(
        Hamiltonian, k=order, compute_eigenvectors=eigenvecs
    )
