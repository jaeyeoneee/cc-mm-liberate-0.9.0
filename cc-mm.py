import numpy as np
import torch
import math
import pickle
import os

from cmt import *
from liberate import fhe
from liberate.fhe import presets
from liberate.fhe.data_struct import data_struct
from scipy.linalg.blas import dgemm
from flint import nmod_mat

def numpy_save(matrix, file_name):
  np.save(file_name, matrix)

def numpy_load(file_name):
  return np.load(file_name)

def ct_matrix_save(matrix, sk, file_name, pk = None):
  with open(file_name, "wb") as f:
    pickle.dump({
      "sk":sk,
      "pk":pk,
      "ct_matrix": matrix
    }, f) 

def ct_matrix_load(file_name):
  with open(file_name, "rb") as f:
    data = pickle.load(f)
  
  return data["sk"], data["pk"], data["ct_matrix"]

def pp_result_save(M00, M01, M10, M11, file_name):
  with open(file_name, "wb") as f:
    pickle.dump({
      "M00":M00,
      "M01":M01,
      "M10":M10,
      "M11":M11
    }, f)
    
def pp_result_load(file_name="test/ppmm.pkl"):
  with open(file_name, "rb") as f:
    data = pickle.load(f)
    
  return data["M00"], data["M01"], data["M10"], data["M11"]

# --------------------------------------------------------------------------------------------------

from copy import deepcopy

def make_all_coeffs_unsigned(cts: list, engine) -> list:
    """
    각 ciphertext (data_struct)의 모든 계수를 unsigned 표현으로 변환
    입력: cts - data_struct 객체 리스트
         engine - liberate fhe engine
    반환: unsigned로 정규화된 data_struct 리스트
    """
    result_cts = []

    for ct in cts:
      if engine.device(ct) != 'cuda:0':
        ct = engine.cuda(ct)
      new_data = []
      for comp in ct.data:  # comp는 [tensor, tensor, ...] 형태
          new_comp = []
          for chunk in comp:
              # 복사해서 처리 (inplace 아님)
              chunk = chunk.clone()
              mult_type = -2 if ct.include_special else -1
              engine.ntt.make_unsigned([chunk], ct.level, mult_type)
              engine.ntt.reduce_2q([chunk], ct.level, mult_type)
              new_comp.append(chunk)
          new_data.append(new_comp)

      new_ct = data_struct(
          data=new_data,
          include_special=ct.include_special,
          ntt_state=ct.ntt_state,
          montgomery_state=ct.montgomery_state,
          origin=ct.origin,
          level=ct.level,
          hash=ct.hash,
          version=ct.version
      )
      result_cts.append(engine.cpu(new_ct))

    return result_cts


def extract_rns_matrices(ct_matrix, ctx_q):
    """
    ct_matrix: length-N 리스트 of data_struct (한 행마다 암호문)
    ctx_q: engine.ctx.q (각 prime modulus 리스트)
    returns: (AU_rns, BU_rns) — 둘 다 리스트 of np.ndarray, 
             각 np.ndarray 는 shape (N, N) 행렬
    """
    N = len(ct_matrix)
    num_mods = ct_matrix[0].data[0][0].size(0)

    # 레벨 p (0 ≤ p < num_mods) 에 대한 A, B행렬을 담을 리스트
    AU_rns = []
    BU_rns = []

    for p in range(num_mods):
        # p번째 modulus 레벨에서 각 행 i의 A-component, B-component 를 뽑아 쌓는다
        B_p = np.vstack([
            ct_matrix[i].data[0][0][p].cpu().numpy()  
            for i in range(N)
        ])
        A_p = np.vstack([
            ct_matrix[i].data[1][0][p].cpu().numpy()  
            for i in range(N)
        ])
        AU_rns.append(A_p)
        BU_rns.append(B_p)

    return AU_rns, BU_rns

def build_cts(engine, rns_mat, template_ct):
    num_mods = len(rns_mat)
    N = rns_mat[0].shape[0]
    dtype = engine.ctx.torch_dtype
    device = torch.device('cpu')  # GPU가 아니라 CPU에 저장

    new_cts = []
    for i in range(N):
        A_parts = [
            torch.tensor(rns_mat[p][i, :], dtype=dtype, device=device)
            for p in range(num_mods)
        ]
        B_parts = [
            torch.zeros_like(A_parts[0])  # poly_len짜리 1D 텐서
            for _ in range(num_mods)
        ]

        A = [torch.stack(A_parts, dim=0)]  # (num_mods, poly_len)
        B = [torch.stack(B_parts, dim=0)]

        ct = data_struct(
            data=[B, A],
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
# ----------------------------------------------------------------------------------------------
  
# def block_rns_matmul(AU_rns, BU_rns, AV_rns, BV_rns, q_list):
  
#   q_len = len(AU_rns)
  
#   M00_rns = []
#   M01_rns = []
#   M10_rns = []
#   M11_rns = []
  
#   for i in range(q_len):
#     qi = q_list[i]
#     AUi, BUi, AVi, BVi = AU_rns[i], BU_rns[i], AV_rns[i], BV_rns[i]
#     M00 = dgemm(alpha = 1.0, a=AUi, b=AVi) % qi
#     M01 = dgemm(alpha = 1.0, a=AUi, b=BVi) % qi
#     M10 = dgemm(alpha = 1.0, a=BUi, b=AVi) % qi 
#     M11 = dgemm(alpha = 1.0, a=BUi, b=BVi) % qi 
    
#     M00_rns.append(M00)
#     M01_rns.append(M01)
#     M10_rns.append(M10)
#     M11_rns.append(M11)
    
#   return M00_rns, M01_rns, M10_rns, M11_rns

# from flint import nmod_mat
# import numpy as np

# def numpy_to_nmod_mat(arr: np.ndarray, q: int) -> nmod_mat:
#     n, m = arr.shape
#     mat = nmod_mat(n, m, q)
#     for i in range(n):
#         for j in range(m):
#             mat[i, j] = int(arr[i, j]) % q
#     return mat

# def nmod_mat_to_numpy(mat: nmod_mat) -> np.ndarray:
#     n, m = mat.nrows(), mat.ncols()
#     return np.array([[int(mat[i, j]) for j in range(m)] for i in range(n)], dtype=np.int64)

# def block_rns_matmul(AU_rns, BU_rns, AV_rns, BV_rns, q_list):
#     """
#     입력 행렬 리스트의 길이에 따라 유효한 RNS modulus만 골라서 FLINT 행렬곱 수행
#     """
#     q_len = len(AU_rns)
#     assert q_len == len(BU_rns) == len(AV_rns) == len(BV_rns), "행렬 레벨 불일치"

#     M00_rns, M01_rns, M10_rns, M11_rns = [], [], [], []

#     for i in range(q_len):
#         qi = q_list[i]  # 현재 레벨에서 사용하는 modulus

#         AUi = numpy_to_nmod_mat(AU_rns[i], qi)
#         BUi = numpy_to_nmod_mat(BU_rns[i], qi)
#         AVi = numpy_to_nmod_mat(AV_rns[i], qi)
#         BVi = numpy_to_nmod_mat(BV_rns[i], qi)

#         M00 = nmod_mat_to_numpy(AUi * AVi)
#         M01 = nmod_mat_to_numpy(AUi * BVi)
#         M10 = nmod_mat_to_numpy(BUi * AVi)
#         M11 = nmod_mat_to_numpy(BUi * BVi)

#         M00_rns.append(M00)
#         M01_rns.append(M01)
#         M10_rns.append(M10)
#         M11_rns.append(M11)

#     return M00_rns, M01_rns, M10_rns, M11_rns

from multiprocessing import Pool
import multiprocessing as mp
from flint import nmod_mat, ctx
import numpy as np

mp.set_start_method('spawn', force=True)

def _nmod_to_np(mat, dtype=np.int64):
    """
    nmod_mat → 2-D np.ndarray   (object 배열로 변환되는 문제 방지)
    """
    r, c = mat.nrows(), mat.ncols()
    out  = np.empty((r, c), dtype=dtype)
    for i in range(r):
        # 한 행씩 list()로 뽑아오면 파이썬 루프 최소화
        out[i, :] = np.fromiter((mat[i, j] for j in range(c)),
                                dtype=dtype, count=c)
    return out
# 전역에 미리 변환할 리스트
flint_A = flint_B = None

def _init_worker(nth, A_list, B_list):
    global flint_A, flint_B
    ctx.threads = nth           # 프로세스당 FLINT 스레드 수 설정
    flint_A = A_list           # fork 이후 공유
    flint_B = B_list

def _worker(idx):
    # idx만 넘어오므로 pickle 비용 거의 없음
    C_nm = flint_A[idx] * flint_B[idx]
    return idx, _nmod_to_np(C_nm)

def pp_mm_all_channels_mp_fast(A_rns, B_rns, q_list,
                               nth_per_proc=64):
    n_mods = len(q_list)
    # 1) 한번만 tolist→nmod_mat
    A_list = [nmod_mat(A.tolist(), q) for A, q in zip(A_rns, q_list)]
    B_list = [nmod_mat(B.tolist(), q) for B, q in zip(B_rns, q_list)]

    # 2) Pool 생성 (프로세스 수 = min(논리코어//nth_per_proc, n_mods))
    max_procs = max(1, os.cpu_count()//nth_per_proc)
    with Pool(processes=min(n_mods, max_procs),
              initializer=_init_worker,
              initargs=(nth_per_proc, A_list, B_list)) as pool:
        results = pool.map(_worker, list(range(n_mods)))

    # 3) 결과 재조합
    C_rns = [None]*n_mods
    for idx, C in results:
        C_rns[idx] = C
    return C_rns


# --------------------------------------------------------------------------------------------------

def ccmm(engine, ctU, ctV,  sk=None):

  """
  
  Algorithm3: CC-MM
  
  """  
  
  N = len(ctU)  
  
  # line 1: transpose ctU first
  # 초기 테스트에서는 저장했던 한 행렬만을 사용하기 때문에 파일에서 불러오는 형식을 사용한다.
  # 이후에 더 적절하게 변경할 필요가 있음 TODO
  if os.path.exists("test/ct_matrix_transpose2.pkl"):
    _, _, ctU_T = ct_matrix_load("test/ct_matrix_transpose2.pkl")
  else:
    ctU_T = c_mt(engine, ctU, sk=sk)
    ct_matrix_save(ctU_T, sk, file_name="test/ct_matrix_transpose2.pkl")

  ctU_T = make_all_coeffs_unsigned(ctU_T, engine)

  for i in range(3):
    print(engine.decode(engine.decrypt(engine.cuda(ctU_T[i]), sk), coeff=True)[:10])

  for i in range(N):
    ctV[i]  = engine.cpu(engine.level_up(engine.cuda(ctV[i]), ctU_T[i].level))

  # line2: pp-mm between ctU transpose and ctV
  if os.path.exists("test/ppmm_7.pkl"):
    M00_rns, M01_rns, M10_rns, M11_rns = pp_result_load("test/ppmm_7.pkl")
  else:
    AU_rns, BU_rns = extract_rns_matrices(ctU_T, engine.ctx.q)
    AV_rns, BV_rns = extract_rns_matrices(ctV, engine.ctx.q)
    M00_rns = pp_mm_all_channels_mp_fast(AU_rns, AV_rns, engine.ctx.q[:len(AU_rns)], 64)
    M01_rns = pp_mm_all_channels_mp_fast(AU_rns, BV_rns, engine.ctx.q[:len(AU_rns)], 64)
    M10_rns = pp_mm_all_channels_mp_fast(BU_rns, AV_rns, engine.ctx.q[:len(AU_rns)], 64)
    M11_rns = pp_mm_all_channels_mp_fast(BU_rns, BV_rns, engine.ctx.q[:len(AU_rns)], 64)
    
    pp_result_save(M00_rns, M01_rns, M10_rns, M11_rns, file_name="test/ppmm_7.pkl")
    
  # M01, M00 to new ciphertext(data_struct)
  M01_rns = build_cts(engine, M01_rns, ctV[0])
  M00_rns = build_cts(engine, M00_rns, ctV[0])
  M10_rns = build_cts(engine, M10_rns, ctV[0])
  M11_rns = build_cts(engine, M11_rns, ctV[0])
  
  # line3: transpose M01
  if os.path.exists("test/ct_matrix_transpose2_7.pkl"):
    _, _, ctU_T_M01 = ct_matrix_load("test/ct_matrix_transpose2_7.pkl")
  else:
    ctU_T_M01 = c_mt(engine, M01_rns, sk=sk)
    ct_matrix_save(ctU_T_M01, sk, file_name="test/ct_matrix_transpose2_7.pkl")  
  
  # line4: transpose M00
  if os.path.exists("test/ct_matrix_transpose3_7.pkl"):
    _, _, ctU_T_M00 = ct_matrix_load("test/ct_matrix_transpose3_7.pkl")
  else:
    ctU_T_M00 = c_mt(engine, M00_rns, sk=sk)
    ct_matrix_save(ctU_T_M00, sk, file_name="test/ct_matrix_transpose3_7.pkl")
  
  ctU_T_M01[0] = engine.cpu(engine.negate(engine.cuda(ctU_T_M01[0])))
  ctU_T_M00[0] = engine.cpu(engine.negate(engine.cuda(ctU_T_M00[0])))
  
  ctU_T_M00 = make_all_coeffs_unsigned(ctU_T_M00, engine)
  ctU_T_M01 = make_all_coeffs_unsigned(ctU_T_M01, engine)
  
  evk = engine.create_evk(sk)
  final_cts = []

  for i in range(N):
    def bring_to_level(ct, target_level):
      while target_level > ct.level:
        ct = engine.rescale(ct)
      return ct
    
    lvl   = max(ctU_T_M00[i].level, ctU_T_M01[i].level, M10_rns[i].level, M11_rns[i].level)
    mtype = -2 if ctU_T_M00[i].include_special else -1

    ctU_T_M00[i] = bring_to_level(ctU_T_M00[i], lvl)
    ctU_T_M01[i] = bring_to_level(ctU_T_M01[i], lvl)
    M10_rns[i] = bring_to_level(M10_rns[i], lvl)
    M11_rns[i] = bring_to_level(M11_rns[i], lvl)
    

    d2 = [ch.clone() for ch in ctU_T_M00[i].data[1]]

    d1 = [ch.clone() for ch in ctU_T_M00[i].data[0]]          
    Acheck = ctU_T_M01[i].data[1]                              
    BAprim = M10_rns[i].data[1]                                
    for k in range(len(d1)):
        d1[k] += Acheck[k] + BAprim[k]

    d0 = [ch.clone() for ch in ctU_T_M01[i].data[0]]          
    BBprim = M11_rns[i].data[0]                                
    for k in range(len(d0)):
        d0[k] += BBprim[k]

    for comp in (d0, d1, d2):
        engine.ntt.mont_enter(comp, lvl, mtype)
        engine.ntt.enter_ntt(comp, lvl)
        engine.ntt.make_unsigned(comp, lvl, mtype)
        engine.ntt.reduce_2q(comp,    lvl, mtype)

    triplet = data_struct(
        data           =(d0, d1, d2),
        include_special=ctU_T_M00[i].include_special,
        ntt_state      =True,
        montgomery_state=True,
        origin         =presets.types.origins["ctt"],
        level          =lvl,
        hash           =ctU_T_M00[i].hash,
        version        =ctU_T_M00[i].version
    )
    ct_W = engine.relinearize(triplet, evk)
    # ct_W = engine.rescale(ct_W)   

    final_cts.append(ct_W)

    if i < 10:
        print(engine.decode(engine.decrypt(ct_W, sk), coeff=True)[:10])
  
  return final_cts


if __name__ == "__main__":
  # # engine 생성
  engine = fhe.ckks_engine(logN= 13, buffer_bit_length = 62, scale_bits = 20, num_special_primes=1, verbose=True)

  sk = engine.create_secret_key()
  pk = engine.create_public_key(sk)

  # # # 처음에 생성했더 같은 matrix를 곱해서 결과가 같은지를 우선 확인해보기!
  # # # N x N 암호문 생성 / 생성한 행렬 저장
  N = engine.ctx.N
  plain_matrix = np.random.randn(N,N)
  for i in range(N):
    plain_matrix[i][0] = 0
  print(plain_matrix)
  
  # 행마다 encrypt하고 cpu에 복사
  ct_matrix = []
  for i in range(N):
    pt = engine.encode(plain_matrix[i], coeff = True)
    ct = engine.encrypt(pt, pk)
    ct_matrix.append(engine.cpu(ct))
  
  numpy_save(plain_matrix, "test/plaintext2.npy")
  ct_matrix_save(ct_matrix, sk, "test/ct_matrix2.pkl", pk=pk)
  
  real_out = plain_matrix @ plain_matrix
  ct_out_matrix = []
  for i in range(N):
    pt = engine.encode(real_out[i], coeff=True)
    ct = engine.encrypt(pt, pk)
    ct_out_matrix.append(engine.cpu(ct))
  ct_matrix_save(ct_out_matrix, sk, "test/ct_out_matrix2.pkl", pk=pk)
  numpy_save(real_out, "test/plain_output2.npy")

  plain = numpy_load("test/plain_output2.npy")
  print("------------plain output-----------")
  for i in range(10):
    print(plain[i][:10])
    
  sk, pk, ct_out = ct_matrix_load("test/ct_out_matrix2.pkl")
  print("-----------   ct output  --------------")
  for i in range(10):
    print(ct_out[i][:10])

  sk, pk, matrix = ct_matrix_load(file_name="test/ct_matrix2.pkl")
  
  ccmm(engine, matrix, matrix, sk)

  # numpy input
  # for i in range(5):
  #   print(plain[i][:5])
    
  # sk, ct_trans = ct_matrix_load("test/ct_matrix_transpose.pkl")
  # for i in range(5):
  #   print(engine.decode(engine.decrypt(engine.cuda(ct_trans[i]), sk), coeff=True)[:5])