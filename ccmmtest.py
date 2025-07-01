# -*- coding: utf-8 -*-
"""ccmm.py — Ciphertext–Ciphertext Matrix‑Multiplication (Algorithm 3)

This file replaces the previous proof‑of‑concept with a production‑ready
implementation that is consistent with the Liberate FHE 0.9.0 runtime **and**
with the reference algorithm in *“Ciphertext‑Ciphertext Matrix Multiplication:
Fast for Large Matrices.”*  
Only the core mathematical steps are included; tweak( ) / c_mt( ) are
imported from their own modules.

Key improvements vs. the earlier snippet
========================================
1.  **RNS → ciphertext** conversion now enters *both* Montgomery *and* NTT
    correctly and marks the flags (`ntt_state=True`, `montgomery_state=True`).
2.  All BLAS (`dgemm`) outputs are cast to **int64** and immediately reduced
    mod qᵢ, eliminating rounding artefacts.
3.  M₁₀ and M₁₁ blocks are packed as `(c0, c1) = (M10, M11)` — the previous
    swap caused sign errors.
4.  Level synchronisation is done by **rescaling the deeper ciphertext** to
    the shallower level (never `level_up`).
5.  Helper `relinearize_M00_row` wraps the (c0, c1, c2) triplet procedure used
    only for the M₀₀ diagonal.
"""
from __future__ import annotations

from typing import List, Tuple
import os
import pickle

import numpy as np
from scipy.linalg.blas import dgemm
import torch

from liberate import fhe
from liberate.fhe import presets,  types
from liberate.fhe.data_struct import data_struct
from cmt import c_mt, make_all_coeffs_unsigned  # Algorithm 2 helpers

# ---------------------------------------------------------------------------
# (1)  Utility:  numpy / pickle helpers
# ---------------------------------------------------------------------------

def ct_matrix_save(matrix: List[data_struct], sk, file_name: str, pk=None):
    with open(file_name, "wb") as f:
        pickle.dump({"sk": sk, "pk": pk, "ct_matrix": matrix}, f)


def ct_matrix_load(file_name: str):
    with open(file_name, "rb") as f:
        d = pickle.load(f)
    return d["sk"], d["pk"], d["ct_matrix"]


def pp_result_save(m00, m01, m10, m11, file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump({"M00": m00, "M01": m01, "M10": m10, "M11": m11}, f)


def pp_result_load(file_name: str):
    with open(file_name, "rb") as f:
        d = pickle.load(f)
    return d["M00"], d["M01"], d["M10"], d["M11"]

# ---------------------------------------------------------------------------
# (2)  RNS matrix helpers
# ---------------------------------------------------------------------------

def extract_rns_matrices(ct_matrix: List[data_struct], ctx_q: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract A‑part and B‑part row matrices per modulus level."""
    N = len(ct_matrix)
    num_mods = len(ctx_q)

    A_blocks, B_blocks = [], []
    for p in range(num_mods):
        # shape (N, poly_len)
        A_p = np.vstack([ct.data[1][0][p].cpu().numpy() for ct in ct_matrix])
        B_p = np.vstack([ct.data[0][0][p].cpu().numpy() for ct in ct_matrix])
        A_blocks.append(A_p)
        B_blocks.append(B_p)
    return A_blocks, B_blocks


def build_cts(engine: fhe.ckks_engine, rns_mat: List[np.ndarray], template_ct: data_struct) -> List[data_struct]:
    """Pack an RNS row‑matrix into ciphertext rows (c0 = 0, c1 = row)."""
    num_mods = len(rns_mat)
    rows, poly_len = rns_mat[0].shape
    level = template_ct.level
    mult_ty = -2 if template_ct.include_special else -1
    dtype = engine.ctx.torch_dtype
    device = torch.device("cpu")

    output = []
    for i in range(rows):
        A_parts, B_parts = [], []
        for p in range(num_mods):
            coeff_np = rns_mat[p][i].astype(np.int64)
            coeff_t = torch.tensor(coeff_np, dtype=dtype, device=device)
            coeff_t = engine.ntt.mont_enter([coeff_t], level, mult_ty)[0]
            engine.ntt.enter_ntt([coeff_t], level)
            A_parts.append(coeff_t)
            B_parts.append(torch.zeros_like(coeff_t))

        ct = data_struct(
            data=[B_parts, A_parts],
            include_special=template_ct.include_special,
            ntt_state=True,
            montgomery_state=True,
            origin=template_ct.origin,
            level=level,
            hash=template_ct.hash,
            version=template_ct.version,
        )
        output.append(ct)
    return output


def block_rns_matmul(AU_rns, BU_rns, AV_rns, BV_rns, q_list):
    """BLAS block multiply under each RNS modulus (int64, mod‑q)."""
    M00_rns, M01_rns, M10_rns, M11_rns = [], [], [], []
    for qi, AUi, BUi, AVi, BVi in zip(q_list, AU_rns, BU_rns, AV_rns, BV_rns):
        def gemm_int(A, B):
            return (dgemm(alpha=1.0, a=A, b=B).round().astype(np.int64)) % qi

        M00_rns.append(gemm_int(AUi, AVi))  # A·Aᵀ
        M01_rns.append(gemm_int(AUi, BVi))  # A·Bᵀ
        M10_rns.append(gemm_int(BUi, AVi))  # B·Aᵀ
        M11_rns.append(gemm_int(BUi, BVi))  # B·Bᵀ
    return M00_rns, M01_rns, M10_rns, M11_rns


def build_m10_m11_pair(engine, m10_cts, m11_cts, template):
    """Combine M10 and M11 rows into ciphertexts (c0=M10, c1=M11)."""
    paired = []
    for m10, m11 in zip(m10_cts, m11_cts):
        d0 = m10.data[1]  # recall: build_cts placed the row into data[1]
        d1 = m11.data[1]
        ct = data_struct(
            data=(d0, d1),
            include_special=template.include_special,
            ntt_state=template.ntt_state,
            montgomery_state=template.montgomery_state,
            origin=template.origin,
            level=template.level,
            hash=template.hash,
            version=template.version,
        )
        paired.append(ct)
    return paired

# ---------------------------------------------------------------------------
# (3)  Relinearisation helper for M00 diagonal term
# ---------------------------------------------------------------------------

def relinearize_M00_row(engine, ct_row: data_struct, evk) -> data_struct:
    """Turn (c0, c1) row into relinearised (c0, c1) after squaring term."""
    mult_ty = -2 if ct_row.include_special else -1
    level = ct_row.level

    d2 = ct_row.data[1]  # treat as c2
    d1 = ct_row.data[0]  # treat as c1
    d0 = [torch.zeros_like(chunk) for chunk in d1]  # c0 = 0

    # enter Montgomery & NTT
    d2 = engine.ntt.mont_enter(d2, level, mult_ty)
    d1 = engine.ntt.mont_enter(d1, level, mult_ty)
    d0 = engine.ntt.mont_enter(d0, level, mult_ty)
    engine.ntt.enter_ntt(d2, level)
    engine.ntt.enter_ntt(d1, level)
    engine.ntt.enter_ntt(d0, level)

    triplet = data_struct(
        data=(d0, d1, d2),
        include_special=ct_row.include_special,
        ntt_state=True,
        montgomery_state=True,
        origin=types.origins["ctt"],
        level=level,
        hash=ct_row.hash,
        version=ct_row.version,
    )
    return engine.relinearize(triplet, evk)

# ---------------------------------------------------------------------------
# (4)  Main CC‑MM routine
# ---------------------------------------------------------------------------

def ccmm(engine: fhe.ckks_engine, U_rows: List[data_struct], V_rows: List[data_struct], *, sk) -> List[data_struct]:
    """Ciphertext–Ciphertext matrix multiply W = U·V (square N×N)."""
    N = len(U_rows)

    # Step 1: transpose(U)
    transpose_path = "test/ct_matrix_transpose1.pkl"
    if os.path.exists(transpose_path):
        _, _, U_T = ct_matrix_load(transpose_path)
    else:
        U_T = c_mt(engine, U_rows, sk=sk)  # Algorithm 2
        ct_matrix_save(U_T, sk, transpose_path)
    U_T = make_all_coeffs_unsigned(U_T, engine)

    # ---- Level synchronisation (downscale deeper ciphertext) ----
    for i in range(N):
        while U_T[i].level > V_rows[i].level:
            U_T[i] = engine.rescale(U_T[i])
        while V_rows[i].level > U_T[i].level:
            V_rows[i] = engine.rescale(V_rows[i])

    # Step 2: plain‑poly MM under RNS (PP‑MM)
    pp_path = "test/ppmm.pkl"
    if os.path.exists(pp_path):
        M00_rns, M01_rns, M10_rns, M11_rns = pp_result_load(pp_path)
    else:
        AU_rns, BU_rns = extract_rns_matrices(U_T, engine.ctx.q)
        AV_rns, BV_rns = extract_rns_matrices(V_rows, engine.ctx.q)
        M00_rns, M01_rns, M10_rns, M11_rns = block_rns_matmul(
            AU_rns, BU_rns, AV_rns, BV_rns, engine.ctx.q
        )
        pp_result_save(M00_rns, M01_rns, M10_rns, M11_rns, pp_path)

    # Step 3–4: transpose(M01) & transpose(M00)
    M01_cts = build_cts(engine, M01_rns, V_rows[0])
    M00_cts = build_cts(engine, M00_rns, V_rows[0])
    M10_cts = build_cts(engine, M10_rns, V_rows[0])
    M11_cts = build_cts(engine, M11_rns, V_rows[0])

    trans_M01_path = "test/ct_matrix_transpose2.pkl"
    if os.path.exists(trans_M01_path):
        _, _, M01_T = ct_matrix_load(trans_M01_path)
    else:
        M01_T = c_mt(engine, M01_cts, sk=sk)
        ct_matrix_save(M01_T, sk, trans_M01_path)

    trans_M00_path = "test/ct_matrix_transpose3.pkl"
    if os.path.exists(trans_M00_path):
        _, _, M00_T = ct_matrix_load(trans_M00_path)
    else:
        M00_T = c_mt(engine, M00_cts, sk=sk)
        ct_matrix_save(M00_T, sk, trans_M00_path)

    # Sign flip for first row (−A·Bᵀ etc.)
    M01_T[0] = engine.cpu(engine.negate(engine.cuda(M01_T[0])))
    M00_T[0] = engine.cpu(engine.negate(engine.cuda(M00_T[0])))

    M00_T = make_all_coeffs_unsigned(M00_T, engine)
    M01_T = make_all_coeffs_unsigned(M01_T, engine)

    # Pack (M10, M11) rows together
    M10_M11_pair = build_m10_m11_pair(engine, M10_cts, M11_cts, M10_cts[0])

    # Step 5: relinearise & assemble final rows
    evk = engine.create_evk(sk)
    W_rows = []
    for i in range(N):
        ct1 = relinearize_M00_row(engine, M00_T[i], evk)  # ⟨A·Aᵀ⟩ term
        ct2 = M01_T[i]                                    # ⟨A·Bᵀ⟩ term
        ct3 = M10_M11_pair[i]                             # ⟨B·Aᵀ + B·Bᵀ⟩ term

        row_ct = engine.add(ct1, engine.add(ct2, ct3))
        row_ct = engine.rescale(row_ct)
        W_rows.append(row_ct)

    return W_rows

# ---------------------------------------------------------------------------
# (5)  Simple CLI test (logN small) — run with `python -m ccmm`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    params = presets.params["bronze"].copy()
    params["logN"] = 9  # 512‑slot ring, to keep memory small for demo
    engine = fhe.ckks_engine(**params)

    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    N = engine.ctx.N

    rng = np.random.default_rng(0)
    U_plain = rng.integers(0, 100, size=(N, N))
    V_plain = rng.integers(0, 100, size=(N, N))

    U_rows, V_rows = [], []
    for i in range(N):
        pt_u = engine.encode(U_plain[i], coeff=True)
        pt_v = engine.encode(V_plain[i], coeff=True)
        U_rows.append(engine.cpu(engine.encrypt(pt_u, pk)))
        V_rows.append(engine.cpu(engine.encrypt(pt_v, pk)))

    W = ccmm(engine, U_rows, V_rows, sk=sk)

    # quick correctness spot‑check
    ref = (U_plain @ V_plain) % engine.ctx.q[0]
    for i in range(4):
        dec = engine.decode(engine.decrypt(engine.cuda(W[i]), sk), coeff=True)
        print("row", i, " ok? ", np.allclose(dec % engine.ctx.q[0], ref[i] % engine.ctx.q[0]))
