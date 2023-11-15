#!/usr/bin/env/python
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

import argparse
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

import netket.experimental as nkx
import numpy as np
import optax

from transformer_LR_WF.hamiltonian import *
from transformer_LR_WF.utils import *
from transformer_LR_WF.vision_transformer import *


def positive_integer(candidate):
    integer_candidate = int(candidate)
    if integer_candidate <= 0:
        raise argparse.ArgumentTypeError("a positive integer was expected")
    return integer_candidate


argument_parser = argparse.ArgumentParser(
    description="run an example calculation for a spin model"
)
argument_parser.add_argument(
    "-d",
    "--diagonalize",
    help="compare the result with the exact diagonalization (up to 20 spins)",
    action="store_true",
)
argument_parser.add_argument(
    "N", type=positive_integer, help="number of spins in the chain"
)
argument_parser.add_argument(
    "alpha", type=float, help="value of the interaction decay parameter"
)
argument_parser.add_argument(
    "J", type=float, help="value of the base spin-spin interaction strength"
)
argument_parser.add_argument("b", type=positive_integer, help="token size")
argument_parser.add_argument(
    "demb", type=positive_integer, help="embedding width"
)
argument_parser.add_argument(
    "h", type=positive_integer, help="number of heads"
)

args = argument_parser.parse_args()

diagonalize = args.diagonalize
N = args.N
alpha = args.alpha
J = args.J
b = args.b
demb = args.demb
h = args.h

if N > 20 and diagonalize:
    sys.exit(
        "Error: exact diagonalization not available for more than 20 spins."
    )
if N % b != 0:
    sys.exit("Error: the token size must be a divisor of the number of spins.")
if demb % h != 0:
    sys.exit(
        "Error: the number of heads must be a divisor of the embedding dimension."
    )


### Creation of the Hilbert space object and the observables ###
hi = nk.hilbert.Spin(s=1 / 2, N=N)

renyi = nkx.observable.Renyi2EntanglementEntropy(
    hi, np.arange(0, N / 2 + 1, dtype=int)
)
mags = sum([(-1) ** i * sigmaz(hi, i) / N for i in range(N)])
magnet = sum([sigmaz(hi, i) / N for i in range(N)])
################################################################

### MC sampling rules ###
rule1 = nk.sampler.rules.LocalRule()
rule2 = InvertMagnetization()
pinvert = 0.25
pflip = 1 - pinvert
sampler = nk.sampler.MetropolisSampler(
    hi, nk.sampler.rules.MultipleRules([rule1, rule2], [pflip, pinvert])
)
################################################################

### Training schedule ###
max_iters = 200
ramp_iter = 50
lrmax = 1.0

lr_schedule = optax.warmup_exponential_decay_schedule(
    0.1,
    peak_value=lrmax,
    warmup_steps=ramp_iter,
    transition_steps=1,
    decay_rate=0.995,
)

optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)

ds_schedule = optax.linear_schedule(1e-2, 1e-4, max_iters)
SR = nk.optimizer.SR(diag_shift=ds_schedule)
################################################################

H = get_Hamiltonian(N=N, J=J, alpha=alpha)

if diagonalize:
    Egs, eigenvec = get_eigvals(Hamiltonian=H, order=1, eigenvecs=True)

model = BatchedSpinViT(
    token_size=b,
    embedding_d=demb,
    n_heads=h,
    n_blocks=1,
    n_ffn_layers=3,
    final_architecture=(5,),
    is_complex=False,
)

vstate = nk.vqs.MCState(
    sampler,
    model,
    n_samples=512,
    n_discard_per_chain=0,
    chunk_size=None,
)

gs = nk.driver.VMC(
    H,
    optimizer,
    variational_state=vstate,
    preconditioner=SR,
)

log = (
    nk.logging.RuntimeLog()
)  # If instead of this logging you insert a string, it will be used as output prefix for a JSON file where the evolution of the energy at each epoch will be stored.
keeper = BestIterKeeper(H, N, 1e-8)

# keeper.filename = 'Somewhere' #It allows you to store the parameters of the model for the state with lowest energy found.

gs.run(n_iter=max_iters, out=log, callback=[keeper.update], show_progress=True)

if diagonalize:
    fidelity = np.abs(keeper.best_state.to_array().conj() @ eigenvec[:, 0])
    rel_err = np.abs((keeper.best_energy - Egs[0]) / Egs[0])
    print(f"Fidelity: {fidelity:.5f}")
    print(f"Relative error in energy: {rel_err:.2E}")

vsc = keeper.vscore
print(f"V-score: {vsc:.2E}")

S = np.real(keeper.best_state.expect(renyi).mean)
print(f"Value for the Renyi-2 entropy: {S:.5f}")

m = np.real(keeper.best_state.expect(magnet).mean)
print(f"Value for the magnetization: {m:.5f}")

ms = np.real(keeper.best_state.expect(mags).mean)
print(f"Value for the staggered magnetization: {ms:.5f}")

fluct = np.real(keeper.best_state.expect(magnet @ magnet).mean)
print(f"Value for the squared magnetization: {fluct:.5f}")

fluct_s = np.real(keeper.best_state.expect(mags @ mags).mean)
print(f"Value for the squared staggered magnetization: {fluct_s:.5f}")
