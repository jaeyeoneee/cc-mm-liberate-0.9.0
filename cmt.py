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

def c_mt(engine, cts, pts_test = None, sk = None):
  
  N = len(cts)
  
  # galois key를 cpu에 넣어야 할 듯?
  # gk = engine.create_galois_key(sk)
  
  # line 1: first tweak
  shifted = [polynomial_X_mult(engine, cts[i], i) for i in range(N)]
  # shifted_test = np.vstack([rotate_with_cyclic_sign(pts_test[i], i) for i in range(N)])
  
  # for i in range(10):
    # print("fhe line 1 -> shifted:", engine.decode(engine.decrypt(engine.cuda(shifted[i]), sk), coeff=True)[:20])
    # print("np line 1 -> shifted:", shifted_test[i][:20])

  aux = tweak(engine, shifted)
  # aux_test = tweak_np(shifted_test)
  
    
  # for i in range(10):
    # print("fhe line 1 -> tweak:", engine.decode(engine.decrypt(engine.cuda(aux[i]), sk), coeff=True)[:20])
    # print("np line 1 -> tweak:", aux_test[i][:20])  
  
  # line 2 ~ 4: automorphism
  inv_N_np = 1/N
  # inv_n = pow(N, -1, reduce(operator.mul, engine.ctx.q, 1))
  aux_p = [None] * N
  # aux_p_np = np.zeros_like(aux_test)
  for j in range(N):
    # line 3: multiply N^-1 to aux
    index = 2*j+1
    inv_index = pow(index, -1, 2*N) // 2
    
    ct_temp = engine.cuda(aux[inv_index])
    ct_scaled = engine.mult_scalar(ct_temp, inv_N_np)
    # np_scaled = inv_N_np * aux_test[inv_index]
    
    # line 4: automorphism
    auto_key = engine.create_automorphism_key(sk, j)
    ct_galois = engine.apply_automorphism(ct_scaled, auto_key)
    
    aux_p[j] = engine.cpu(ct_galois)
    # aux_p_np[j] = automorphism_np(np_scaled, index)

  # # print("line 2~4 -> aux_p:", aux_p)
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
    ct_pp_neg = engine.cpu(engine.negate(ct_pp_mult_X))
    
    ct_out[index_out] = ct_pp_neg
    # ct_out_np[index_out] = -1 * rotate_with_cyclic_sign(ct_pp_np[index], mult)
    
  
  for i in range(10):
    print("fhe line 8->", engine.decode(engine.decrypt(engine.cuda(ct_out[i]), sk), coeff=True)[:20])
  #   # print("np line 8->", ct_out_np[i][:20])
  
  return ct_out

if __name__ == "__main__":
  # # numpy test for c-mt algorithm
  # print("------ c_mt_np test----------")
  # N = 2**14 # Ring dimension
  # n = 2**14 # the number of ciphertexts
  
  # input = np.zeros((n, N))
  # for i in range(n):
  #   for j in range(N):
  #     input[i][j] = j
  
  # # c_mt_rs = c_mt_np(input)
  # # # print("input:", input)
  # # # print("c_mt_rs:", c_mt_rs)
  
  # liberate c-mt algorithm test
  # print("------ c_mt liberate fhe test ----------")
  params = presets.params["bronze"]
  # params["logN"] = 10
  engine = fhe.ckks_engine(**params)
  # engine = fhe.ckks_engine(logN= 13, buffer_bit_length = 62, scale_bits = 40, num_special_primes=1, verbose=True)
  sk = engine.create_secret_key()
  pk = engine.create_public_key(sk)

  N = engine.ctx.N
  # Bronze 사이즈에 대해서 대략 41기가 정도 크기 필요 cpu ram    
  
  cts = []
  pts_test = [np.arange(N) % 10 for _ in range(N)]
  # pts_test = np.load("test/plain_matrix.npy")
  print("input[0]:", pts_test[0])
  for i in range(N):
    pts = engine.encode(pts_test[i], coeff=True)
    ct = engine.encrypt(pts, pk)
    cts.append(engine.cpu(ct))
  
  import time
  
  t0 = time.perf_counter()
  
  c_mt(engine, cts, pts_test, sk = sk)
  
  elapsed = time.perf_counter() - t0
  
  print(elapsed)
  
  # # automorphsim_test
  # print("------ automorphism_np test -------")
  # vec = np.array([1, 2, 3, 4])
  # print("vec:", vec)
  # k = 5
  # print("k:", k)
  # auto_rs = automorphism_np(vec, k)
  # print("automorphism_np result:", auto_rs)
  
  # liberate automorphism test
  # print("----------automorphism test liberate--------")
  # params = presets.params["bronze"]
  # engine = fhe.ckks_engine(**params)

  # sk = engine.create_secret_key()
  # pk = engine.create_public_key(sk)

  # input = np.arange(engine.ctx.N)
  # ct = engine.encrypt(engine.encode(input, coeff=True), pk)
  
  # rotk = engine.create_automorphism_key(sk, 2)
  # rotated_ct = engine.apply_automorphism(ct, rotk)
  
  # print("decrypted rot X^3:", engine.decode(engine.decrypt(rotated_ct, sk), coeff=True)[:10])
  # print("automorphism X^3 np:", automorphism_np(input, 5)[:10])

  #  Test for rotate in encdec.py

  # # N = 16
  # torch.manual_seed(0)
  # vec_np = np.random.randint(-10, 10, size=N)
  # vec_t = torch.tensor(vec_np, dtype=torch.int64)

  # ks = []
  # for i in range(1*N):
  #   ks.append(i)
  # results = []

  # for k in ks:
  #   rot_t = rotate_poly(vec_t, k)
  #   rot_np_from_t = rot_t.numpy()
  #   rot_np_gt = automorphism_np(vec_np, 2*k+1)
  #   equal = np.array_equal(rot_np_from_t, rot_np_gt)
  #   results.append({
  #       'k': k,
  #       'rotate_poly': rot_np_from_t.tolist(),
  #       'automorphism_np': rot_np_gt.tolist(),
  #       'match': equal
  #   })
  #   print(equal)
  # # print(results)
  
  # #  modular test for c-mt algorithm line 8
  # N = engine.ctx.N
  # for j in range(1, N+1):
  #   print(pow(N-j, 1, N))
  
  