import numpy as np
import torch
import math
import os

from liberate import fhe
from liberate.fhe import presets
from liberate.fhe.data_struct import data_struct
from scipy.linalg.blas import dgemm
from flint import nmod_mat, ctx as flint_ctx
from liberate.fhe.presets import types, errors


"""
기본적으로 암호문은 모두 cpu에 있다고 생각하기!
"""

class CCMM:
  
  def __init__(self):
    # engine 
    # self.params = presets.params["bronze"]
    # self.engine = fhe.ckks_engine(**self.params)
    self.engine = fhe.ckks_engine(logN= 13, buffer_bit_length = 62, scale_bits = 25, num_special_primes=1, verbose=True)
    
    # key
    self.sk = self.engine.create_secret_key()
    self.pk = self.engine.create_public_key(self.sk)
    self.evk = self.engine.create_evk(self.sk)

    # params
    self.slot_size = self.engine.ctx.N
    self.num_levels = self.engine.num_levels
    self.scale = self.engine.scale
    self.q = self.engine.ctx.q
  
  def engine_info(self):
    
    print()
    print("--------------engine-info-------------------")
    print(f"num_levels: {self.num_levels}")
    print(f"num_slots: {self.slot_size}")
    print(f"scale: {self.scale}")
    print(f"q: {self.q}")
    print("--------------------------------------------")
    
  def encode(self, matrix:np.ndarray):
    """
    인풋으로 넘파이 행렬을 받아서 계수 인코딩으로 암호화한다. 암호문 리스트를 반환한다.
    
    Args:
        matrix (_type_): (self.slot_size, self.slot_siez) 크기의 넘파이 배열
    
    Returns:
        matrix을 행별로 암호화한 리스트
    """
    r, c = matrix.shape
    n = self.slot_size

    # 2차원 확인
    if matrix.ndim != 2:
      raise ValueError(f"matrix must to be 2D, got ndim={matrix.ndim}")
    
    # 행렬 크기 확인
    if (r, c) != (n, n):
      raise ValueError(
        f"matrix shape must be ({n}, {n}) to match slot_size, but got {matrix.shape}"
      )
      
    cts = []
    
    for i in range(n):
      row = matrix[i]
      encoded_row = self.engine.encode(row, coeff=True)
      encrypted_row = self.engine.encrypt(encoded_row, self.pk)
      cpu_row = self.engine.cpu(encrypted_row)
      cts.append(cpu_row)

    return cts
  
  def decode(self, cts):
    """
    인풋으로 암호문 리스트를 받아서 넘파이 행렬로 반환해준다.
    
    Args:
      cts: liberate fhe로 암호화된 암호문을 가지는 list
    Returns:
      cts를 복호화한 numpy 행렬 배열
    """
    
    n = self.slot_size
    engine = self.engine  
    
    if len(cts) != n:
      raise ValueError(f"cts length must be {n}, got{len(cts)}")
    
    rows = []
    decoded_row = None
    for ct in cts:
      
      ct = engine.cuda(ct)
      
      decrypted_row = engine.decrypt(ct, self.sk)
      
      decoded_row = engine.decode(decrypted_row, coeff=True)
    
      rows.append(decoded_row)  
  
    mat = np.vstack(rows)
    
    if mat.shape != (n, n):
      raise ValueError(f"decoded matrix shape {mat.shape} != ({n}, {n})")

    return mat
  
  # -------------------------
  # TWEAK
  # -------------------------
  
  def rotate_with_cyclic_sign_np(self, vector:np.ndarray, shift:int)->np.ndarray:
    
    N = vector.shape[-1]
    r = shift // N
    s = shift % N
    
    out = np.roll(vector, s)
    
    if s > 0:
      out[:s] *= -1
    
    if (r % 2) == 1:
      out *= -1
      
    return out
    
  
  def tweak_np(self, matrix: np.ndarray) -> np.ndarray:
    """
    Algorithm 1 TWEAK의 Numpy version.
    """
    n, N = matrix.shape

    if n == 1:
      return matrix.copy()
    
    result_blocks = [None] * n
    result_blocks[0] = matrix[0].copy()
    
    for ell in range(int(np.log2(n))):
      two_ell = 2 ** ell
      block = n // (2*two_ell)
      
      aux_inputs = [matrix[(2*j+1)*block] for j in range(two_ell)]
      aux = self.tweak_np(np.vstack(aux_inputs))
      
      for j in range(two_ell):
        shift = (N//(2**ell))*j
        rotated = self.rotate_with_cyclic_sign_np(aux[j], shift)
        
        result_blocks[j+two_ell] = result_blocks[j] - rotated
        result_blocks[j]         = result_blocks[j] + rotated
        
    return np.vstack(result_blocks)

  def rotate_with_cyclic_sign(self, ct, shift):
    """
    multiply ct(X) polynomial by x^i
    """
    
    r = shift // self.slot_size
    s = shift % self.slot_size
    
    ct = self.engine.cuda(ct)
    shifted_data = []
    
    def normalize_unsigned(chunk: torch.Tensor, level, include_special):
      mult_type = -2 if include_special else -1
      self.engine.ntt.make_unsigned([chunk], level, mult_type)
      self.engine.ntt.reduce_2q(    [chunk], level, mult_type)
      return chunk
    
    for comp in ct.data:
      shifted_comp = []
      for chunk in comp:
        rolled = torch.roll(chunk, shifts=shift, dims=-1)
        if s != 0:
          rolled[..., :s] *= -1
        if (r % 2) == 1:
          rolled *= -1
        rolled = normalize_unsigned(rolled, ct.level, ct.include_special)
        shifted_comp.append(rolled)
      shifted_data.append(shifted_comp)
    
    return self.engine.cpu(data_struct(
        data=shifted_data,
        include_special=ct.include_special,
        ntt_state=ct.ntt_state,
        montgomery_state=ct.montgomery_state,
        origin=ct.origin,
        level=ct.level,
        hash=ct.hash,
        version=ct.version
    ))
        
        
  def tweak(self, cts):
    
    n = len(cts)
    
    if n == 1:
      return cts
    
    ct_p = [None] * n
    ct_p[0] = cts[0]
    
    for l in range(0, int(np.log2(n))):
      pow2 = 2**l
      block = n // (2*pow2)
      
      aux_cts = [cts[(2*j+1)*block] for j in range(pow2)]
      aux = self.tweak(aux_cts)
    
      for j in range(0, pow2):
        shift_k = (self.slot_size // (2**l)) * j
        ct_rot = self.rotate_with_cyclic_sign(aux[j], shift_k)
        ct_rot = self.engine.cuda(ct_rot)
        
        ct_pj = self.engine.cuda(ct_p[j])
        
        ct_sub = self.engine.sub(ct_pj, ct_rot)
        ct_add = self.engine.add(ct_pj, ct_rot)
        
        ct_p[j+pow2] = self.engine.cpu(ct_sub)
        ct_p[j] = self.engine.cpu(ct_add)
      
    return ct_p

  # -------------------------------
  # Transpose
  # -------------------------------

  def automorphism_np(self, vector, shift):
    
    N = vector.shape[-1]
    result = np.zeros_like(vector)
    
    for i, c in enumerate(vector):
      auto = (i*shift) % (2*N)
      sign = -1 if (auto//N) % 2 == 1 else 1
      index = auto - N if auto >= N else auto
      result[index] = sign * c
    
    return result

  def cmt_np(self, matrix):
    """
    Algorithm2: C-MT(transpose) algorithm for numpy test
    matrix "should" be square matrix NxN
    """
    N = matrix.shape[0]
    
    shifted = np.vstack([self.rotate_with_cyclic_sign_np(matrix[i], i) for i in range(N)])
    aux = self.tweak_np(shifted)
    
    inv_N = 1/N
    aux_p = np.zeros_like(aux, dtype=np.float64)
    for j in range(N):
      index = (2 *j + 1)
      inv= pow(index, -1, 2*N)
      inv_index = (inv-1)//2
      aux_p[j] = inv_N * aux[inv_index]
      aux_p[j] = self.automorphism_np(aux_p[j], index)
    
    ct_pp = self.tweak_np(aux_p)
    
    ct_out = np.zeros_like(ct_pp, dtype=np.float64)
    for j in range (1, N+1):
      if j == N:
        ct_out[0] = self.rotate_with_cyclic_sign_np(ct_pp[(N-j) % N], N-j)
      else: 
        ct_out[j % N] = -1 * self.rotate_with_cyclic_sign_np(ct_pp[(N-j) % N], N-j)
    
    return ct_out
  
  def cmt(self, cts):
    
    N = self.slot_size
    
    shifted = [self.rotate_with_cyclic_sign(cts[i], i) for i in range(N)]
    aux = self.tweak(shifted)
    
    aux_p = [None] * N
    for j in range(N):
      index = 2*j+1
      inv = pow(index, -1, 2*N)
      inv_index = (inv)//2
      
      ct_temp = self.engine.cuda(aux[inv_index])
      ct_scaled = self.engine.mult_invN_coeff(ct_temp)
      
      auto_key = self.engine.create_automorphism_key(self.sk, j)
      ct_galois = self.engine.apply_automorphism(ct_scaled, auto_key)
      
      aux_p[j] = self.engine.cpu(ct_galois)

    ct_pp = self.tweak(aux_p)
    ct_out = [None] * N
    for j in range(1, N+1):
      mult = N - j
      index = mult % N
      index_out = j % N
    
      ct_pp_mult_X = self.engine.cuda(self.rotate_with_cyclic_sign(ct_pp[index], mult))
      
      if (j % N != 0):
        ct_pp_neg = self.engine.negate_coeff(ct_pp_mult_X)
      else: 
        ct_pp_neg = ct_pp_mult_X
      
      ct_pp_neg = self.engine.cpu(ct_pp_neg)
    
      ct_out[index_out] = ct_pp_neg
        
    return ct_out
  
  # -------------------------------
  # CC-MM
  # -------------------------------
  
  def make_level1_matrix(self, cts):
    
    N = self.slot_size
    max_level = self.engine.num_levels - 2

    new_cts = []    
    for i in range(N):
      cuda_ct = self.engine.cuda(cts[i])
      new_ct = self.engine.level_up(cuda_ct, max_level)
      gpu_ct = self.engine.cpu(new_ct)
      new_cts.append(gpu_ct)
  
    return new_cts
  

  def extract_matrices_from_cts(self, cts):
    
    N = self.slot_size
    num_rns = cts[0].data[0][0].size(0)
    
    A = []
    B = []

    for p in range(num_rns):
      A_p = np.vstack([
        cts[i].data[1][0][p].cpu().numpy()
        for i in range(N)
      ])
      B_p = np.vstack([
        cts[i].data[0][0][p].cpu().numpy()
        for i in range(N)
      ])
      
      A.append(A_p)
      B.append(B_p)
      
    return A, B

  def build_cts(self, A_or_None, B_or_None, template_ct):
    """
    A, B로부터 다시 암호문을 만들어주는 함수. 둘 중 하나는 None이 아니어야 한다. 나중에 수정 필요~!
    """
    
    if A_or_None is None:
      num_mods = len(B_or_None)
    else:
      num_mods = len(A_or_None)
      
    N = self.slot_size
    dtype = self.engine.ctx.torch_dtype
    device = torch.device('cpu')  # GPU가 아니라 CPU에 저장

    if A_or_None is None:
      A = [np.zeros((N, N), dtype=np.int64) for _ in range(num_mods)]
    else:
      A = A_or_None
    
    if B_or_None is None:
      B = [np.zeros((N, N), dtype=np.int64) for _ in range(num_mods)]
    else:
      B = B_or_None

    new_cts = []
    for i in range(N):
        A_parts = [
            torch.tensor(A[p][i, :], dtype=dtype, device=device)
            for p in range(num_mods)
        ]
        B_parts = [
            torch.tensor(B[p][i, :], dtype=dtype, device=device)  
            for p in range(num_mods)
        ]

        A_new = [torch.stack(A_parts, dim=0)]  
        B_new = [torch.stack(B_parts, dim=0)]

        ct = data_struct(
            data=[B_new, A_new],
            include_special=template_ct.include_special,
            ntt_state=template_ct.ntt_state,
            montgomery_state=template_ct.montgomery_state,
            origin=template_ct.origin,
            level=template_ct.level,
            hash=template_ct.hash,
            version=template_ct.version
        )

        new_cts.append(ct)

    return new_cts
  
  def build_triple_cts_relin(self, A, B_or_None, C_or_None, template_ct):

    assert A is not None
    eng   = self.engine
    N     = self.slot_size
    lvl   = template_ct.level
    incsp = template_ct.include_special
    mtype = -2 if incsp else -1
    dtype = eng.ctx.torch_dtype

    dest_lists = eng.ntt.p.destination_arrays_with_special[lvl] if incsp \
                 else eng.ntt.p.destination_arrays[lvl]
    flat = [x for group in dest_lists for x in group]
    min_abs = min(flat); max_abs = max(flat)
    rows_total = max_abs - min_abs + 1

    L = len(A)
    B = B_or_None if B_or_None is not None else [np.zeros((N,N), dtype=np.int64) for _ in range(L)]
    C = C_or_None if C_or_None is not None else [np.zeros((N,N), dtype=np.int64) for _ in range(L)]

    sorted_abs = sorted(set(flat))
    assert len(sorted_abs) == L, "A/B/C 길이와 활성 림 개수가 일치해야 합니다."
    abs2pos = {abs_idx: pos for pos, abs_idx in enumerate(sorted_abs)}

    out = []
    for i in range(N):
        A_cpu = torch.zeros((rows_total, N), dtype=dtype, device="cpu")
        B_cpu = torch.zeros((rows_total, N), dtype=dtype, device="cpu")
        C_cpu = torch.zeros((rows_total, N), dtype=dtype, device="cpu")

        for abs_idx in flat:
            r = abs_idx - min_abs
            p = abs2pos[abs_idx]
            A_cpu[r, :] = torch.from_numpy(A[p][i, :])
            B_cpu[r, :] = torch.from_numpy(B[p][i, :])
            C_cpu[r, :] = torch.from_numpy(C[p][i, :])

        ct_cpu = data_struct(
            data=[[C_cpu], [B_cpu], [A_cpu]],        
            include_special=incsp,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ctt"], 
            level=lvl,
            hash=template_ct.hash,
            version=template_ct.version
        )

        ct_gpu = eng.cuda(ct_cpu)

        for comp in ct_gpu.data:
            eng.ntt.make_unsigned(comp, lvl, mtype)
            eng.ntt.reduce_2q(     comp, lvl, mtype)
            eng.ntt.mont_enter(    comp, lvl, mtype)
            eng.ntt.enter_ntt(     comp, lvl)       
            eng.ntt.make_unsigned( comp, lvl, mtype)
            eng.ntt.reduce_2q(     comp, lvl, mtype)

        ct_gpu = data_struct(
          data=ct_gpu.data,                           
          include_special=ct_gpu.include_special,
          ntt_state=True,                            
          montgomery_state=True,                      
          origin=ct_gpu.origin,                       
          level=ct_gpu.level,
          hash=ct_gpu.hash,
          version=ct_gpu.version,
        )

        ct_rl = eng.relinearize(ct_gpu, self.evk)

        out.append(eng.cpu(ct_rl))

    return out

  def blas_mat_mult(self, A_list, B_list, level, nth_per_proc=64):

    L = len(A_list)
    out = [None]*L

    dest = self.engine.ntt.p.destination_arrays[level][0]
    q_list = self.q[dest[0]:dest[-1]+1]

    flint_ctx.threads = nth_per_proc
    for i in range(L):
        qi = int(q_list[i])
        Ai = nmod_mat(A_list[i].tolist(), qi)
        Bi = nmod_mat(B_list[i].tolist(), qi)
        Ci = Ai * Bi
        r, c = Ci.nrows(), Ci.ncols()
        Cnp = np.empty((r, c), dtype=np.int64)
        for rr in range(r):
            Cnp[rr, :] = [int(Ci[rr, cc]) for cc in range(c)]
        out[i] = Cnp
    return out
  
  def rescale_matrix(self, cts):
    
    new_cts = []
    for ct in cts:
      ct_cuda = self.engine.cuda(ct)
      ct_rescaled = self.engine.rescale(ct_cuda)
      ct_cpu = self.engine.cpu(ct_rescaled)
      new_cts.append(ct_cpu)
    return new_cts

  def add_matrix(self, ctu, ctv):
    
    new_cts = []
    for i in range(len(ctu)):
      ct1 = ctu[i]
      ct2 = ctv[i]
      ct1_cuda = self.engine.cuda(ct1)
      ct2_cuda = self.engine.cuda(ct2)
      added = self.engine.add(ct1_cuda, ct2_cuda)
      added_cpu = self.engine.cpu(added)
      new_cts.append(added_cpu)
    return new_cts    
  
  def ccmm(self, ctu, ctv):
    
    print("level down-preprocessing")
    ctu = self.make_level1_matrix(ctu)
    ctv = self.make_level1_matrix(ctv)
    
    print("first transpose")
    ctu_T = self.cmt(ctu)    
    
    print("extract A, B-preprocessing for blas ppmm")
    AU, BU = self.extract_matrices_from_cts(ctu_T)
    AV, BV = self.extract_matrices_from_cts(ctv)
    
    print("blas mat mult")
    level = ctu[0].level
    M00 = self.blas_mat_mult(AU, AV, level)
    M01 = self.blas_mat_mult(AU, BV, level)
    M10 = self.blas_mat_mult(BU, AV, level)
    M11 = self.blas_mat_mult(BU, BV, level)
    
    print("build M00 and M01")
    template_ct = ctu[0]
    ct_M01 = self.build_cts(M01, None, template_ct)
    ct_M00 = self.build_cts(M00, None, template_ct)
    
    print("second transpose M00 and M01")
    ct_M01_T = self.cmt(ct_M01)
    ct_M00_T = self.cmt(ct_M00)
    
    print("extract matrices")
    A_down, B_down = self.extract_matrices_from_cts(ct_M01_T)
    A_up, B_up = self.extract_matrices_from_cts(ct_M00_T)
    
    print("relinearization")
    ct_M00_T_rescaled = self.build_triple_cts_relin(A_up, B_up, None, template_ct)
    A_up, B_up = self.extract_matrices_from_cts(ct_M00_T_rescaled)
    
    print("build cts for addition")
    ct1 = self.build_cts(A_up, B_up)
    ct2 = self.build_cts(A_down, B_down)
    ct3 = self.build_cts(M10, M11)
    
    print("add ciphertexts matrices")
    added = self.add_matrix(ct1, ct2)
    added = self.add_matrix(added, ct3)
    
    print("rescale")
    rescaled = self.rescale_matrix(added)
    
    return rescaled