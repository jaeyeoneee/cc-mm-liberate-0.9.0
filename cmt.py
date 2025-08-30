import numpy as np
import torch
import math
import operator

from tweak import *
from liberate import fhe
from liberate.fhe import presets
from liberate.fhe.data_struct import data_struct
from functools import reduce
from liberate.fhe.encdec.encdec import *



def automorphism_np(vec, k):
  """
  numpy automorphism implementationf for c_mt_np
  vec: numpy array of shape (N,)
  k: integer, the automorphism index
  """
  N = vec.shape[-1]
  rs = np.zeros_like(vec)
  for i, c in enumerate(vec):
    auto = (i*k) % (2*N)
    sign = -1 if (auto // N) % 2 == 1 else 1
    index = auto - N if auto >= N else auto
    rs[index] = sign * c
  return rs

def c_mt_np(matrix):
  """
  Algorithm2: C-MT(transpose) algorithm for numpy test
  matrix "should" be square matrix NXN
  """
  N = matrix.shape[0]
  
  # line 1: first tweak
  shifted = np.vstack([rotate_with_cyclic_sign(matrix[i], i) for i in range(N)])
  # print("line 1 -> shifted:", shifted)
  aux = tweak_np(shifted)
  print("line 1 -> tweak_np:", aux)
  
  # line 2 ~ 4: automorphism
  inv_N = 1/N
  aux_p = np.zeros_like(aux)
  for j in range(N):
    # line 3: multiply N^-1 to aux
    index = 2 * j + 1
    inv_index = pow(index, -1, 2*N) // 2
    aux_p[j] = inv_N * aux[inv_index]
    # line 4: automorphism
    aux_p[j] = automorphism_np(aux_p[j], index)
  print("line 2~4 -> aux_p:", aux_p)

  # line 6: second tweak
  ct_pp = tweak_np(aux_p)
  print("line 6 -> tweak_np:", ct_pp)
  
  # line 7~8: final polynomial shift by -X^(N-j)
  ct_out = np.zeros_like(ct_pp)
  for j in range(1, N+1):
    index = (N-j) % N
    ct_out[j%N] = -1 * rotate_with_cyclic_sign(ct_pp[index], N-j)
  print("line 7~8 -> ct_out:", ct_out)
  
  return ct_out

def mul_by_invN_mod(engine, ct):
    # ct: data_struct (보통 NTT+Montgomery 상태)
    N = engine.ctx.N
    level = ct.level
    include_special = ct.include_special
    mult_type = -2 if include_special else -1

    # 1) (선택) 몽고메리 해제해서 깔끔하게 처리
    #    라이브러리 내에 from_montgomery가 없으면 make_unsigned/reduce_2q만으로도 괜찮은 경우가 많음
    #    여기선 정규화만 먼저 해줄게
    for comp in ct.data:
        engine.ntt.make_unsigned(comp, level, mult_type)
        engine.ntt.reduce_2q(    comp, level, mult_type)

    # 2) 각 limb에 대해 q_i, invN_i를 구해 모듈러 곱
    #    ct.data의 shape이 [2 components][num_limbs][N] 라는 가정
    for comp in ct.data:                # c0, c1
        for limb_idx, chunk in enumerate(comp):   # 각 q_i 잔여
            q_i = engine.ctx.q[level+limb_idx]   # <- 네 엔진에서 q 접근 방법에 맞춰 수정
            invN = pow(N, -1, int(q_i))
            # invN = 1/N
            # 텐서에 스칼라 모듈러 곱: (chunk * invN) % q_i
            # dtype이 torch.int64라면 아래처럼:
            chunk.mul_(invN)        # 곱
            engine.ntt.reduce_2q([chunk], level, mult_type)  # mod q 정리

    # 3) (선택) 필요 시 몽고메리로 복귀
    #    보통 다음 연산이 기대하는 도메인/표시에 맞춰 플래그 유지
    return ct


def c_mt(engine, cts, pts_test = None, sk = None):
  
  N = len(cts)
  
  # galois key를 cpu에 넣어야 할 듯?
  # gk = engine.create_galois_key(sk)
  
  # line 1: first tweak
  shifted = [polynomial_X_mult(engine, cts[i], i) for i in range(N)]
  shifted_test = np.vstack([rotate_with_cyclic_sign(pts_test[i], i) for i in range(N)])
  
  # for i in range(10):
  #   print("fhe line 1 -> shifted:", engine.decode(engine.decrypt(engine.cuda(shifted[i]), sk), coeff=True)[:20])
  #   print("np line 1 -> shifted:", shifted_test[i][:20])

  aux = tweak(engine, shifted)
  aux_test = tweak_np(shifted_test)
  
    
  # for i in range(10):
  #   print("fhe line 1 -> tweak:", engine.decode(engine.decrypt(engine.cuda(aux[i]), sk), coeff=True)[:20])
  #   print("np line 1 -> tweak:", aux_test[i][:20])  
  
  # line 2 ~ 4: automorphism
  inv_N_np = 1/N
  # inv_n = pow(N, -1, reduce(operator.mul, engine.ctx.q, 1))
  aux_p = [None] * N
  aux_p_np = np.zeros_like(aux_test)
  for j in range(N):
    # line 3: multiply N^-1 to aux
    index = 2*j+1
    inv_index = pow(index, -1, 2*N) // 2
    
    ct_temp = engine.cuda(aux[inv_index])
    # print(inv_N_np)
    ct_scaled = engine.mult_invN_coeff(ct_temp)
    # ct_scaled = engine.mult_scalar(ct_temp, inv_N_np)
    np_scaled = inv_N_np * aux_test[inv_index]

    print("fhe line 3->", engine.decode(engine.decrypt(ct_scaled, sk), coeff=True)[:20])
    print("np line 3->", np_scaled[:20])
    
    # line 4: automorphism
    auto_key = engine.create_automorphism_key(sk, j)
    ct_galois = engine.apply_automorphism(ct_scaled, auto_key)
    
    aux_p[j] = engine.cpu(ct_galois)
    aux_p_np[j] = automorphism_np(np_scaled, index)

  # print("line 2~4 -> aux_p:", aux_p)
  # for i in range(10):
  #   print("fhe line 3->", engine.decode(engine.decrypt(engine.cuda(aux_p[i]), sk), coeff=True)[:20])
  #   print("np line 3->", aux_p_np[i][:20])

  # line 6: second tweak
  ct_pp = tweak(engine, aux_p)
  # ct_pp_np = tweak_np(aux_p_np)
  
  # for i in range(10):
  #   print("fhe line 6->", engine.decode(engine.decrypt(engine.cuda(ct_pp[i]), sk), coeff=True)[:20])
  #   print("np line 6->", ct_pp_np[i][:20])
  
  # line 7~8: final polynomial shift by -X^(N-j)
  ct_out = [None] * N
  # ct_out_np = np.zeros_like(ct_pp_np)
  for j in range(1, N+1):
    mult = N - j
    index = mult % N
    index_out = j % N
    
    ct_pp_mult_X = engine.cuda(polynomial_X_mult(engine, ct_pp[index], mult))
    ct_pp_neg = engine.cpu(engine.negate_coeff(ct_pp_mult_X))
    
    ct_out[index_out] = ct_pp_neg
    # ct_out_np[index_out] = -1 * rotate_with_cyclic_sign(ct_pp_np[index], mult)
    
  
  for i in range(10):
    print("fhe line 8->", engine.decode(engine.decrypt(engine.cuda(ct_out[i]), sk), coeff=True)[:20])
  #   # print("np line 8->", ct_out_np[i][:20])
  
  return ct_out

if __name__ == "__main__":
  # liberate c-mt algorithm test
  # print("------ c_mt liberate fhe test ----------")
  # params = presets.params["bronze"]
  # engine = fhe.ckks_engine(**params)
  engine = fhe.ckks_engine(logN= 13, buffer_bit_length = 62, scale_bits = 30, num_special_primes=1, verbose=True)
  sk = engine.create_secret_key()
  pk = engine.create_public_key(sk)

  N = engine.ctx.N
  
  rng = np.random.default_rng(0)
  plain = rng.random((N, N), dtype=np.float64)  # [0,1) 범위

  import time

  # 각 행을 계수 인코딩 → 암호화
  t0 = time.perf_counter()
  cts = []
  for i in range(N):
      pt = engine.encode(plain[i], coeff=True)
      ct = engine.encrypt(pt, pk)
      cts.append(engine.cpu(ct))
  t_enc = time.perf_counter() - t0
  print(f"[enc] encoded+encrypted {N} rows in {t_enc:.2f}s")

  # C-MT 실행
  t0 = time.perf_counter()
  ct_out = c_mt(engine, cts, plain,  sk=sk)  # pts_test는 없어도 됩니다
  t_cmt = time.perf_counter() - t0
  print(f"[c_mt] finished in {t_cmt:.2f}s")

  # 결과 복호화/디코드해서 행렬로 복원
  t0 = time.perf_counter()
  rec_rows = []
  for j in range(N):
      dec = engine.decrypt(engine.cuda(ct_out[j]), sk)
      rec = engine.decode(dec, coeff=True)          # shape: (N,)
      rec_rows.append(rec.astype(np.float64))
  rec_mat = np.vstack(rec_rows)                     # shape: (N, N)
  t_dec = time.perf_counter() - t0
  print(f"[dec] decrypted+decoded in {t_dec:.2f}s")

  # 정답과 비교 (Transpose)
  target = plain.T
  diff = rec_mat - target
  mae = np.mean(np.abs(diff))
  maxe = np.max(np.abs(diff))
  print(f"[err] MAE={mae:.6e}, MAX={maxe:.6e}")
  
  print(f"original: {plain}")
  print(f"cmt: {rec_mat}")