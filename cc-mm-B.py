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
from flint import nmod_mat, flintlib, ctx
from concurrent.futures import ThreadPoolExecutor

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

# --------------------------------------------------------------------
# ① 안전한 RNS 블록 행렬곱 – 싱글 채널
# --------------------------------------------------------------------
from flint import nmod_mat
import numpy as np
from math import ceil, log2
from concurrent.futures import ThreadPoolExecutor, as_completed

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
  
def _block_matmul_mod(A_np: np.ndarray,
                      B_np: np.ndarray,
                      modulus: int,
                      block_exp: int = 10):
    """
    한 채널 행렬곱 C = A·B (mod p).
    - 블록 크기 = 2**block_exp (기본 1024 ⇒ 32 MiB 블록 3개 이하)
    - flint.nmod_mat 를 이용해 곱셈 - 덧셈 단계마다 모듈러 연산.
    반환값: np.ndarray dtype=int64, shape (N,N)
    """
    N          = A_np.shape[0]
    blk        = 1 << block_exp           # 2**block_exp
    n_blocks   = ceil(N / blk)
    C_np       = np.zeros((N, N), dtype=np.int64)

    for bi in range(n_blocks):
        i0, i1 = bi*blk, min((bi+1)*blk, N)
        for bj in range(n_blocks):
            j0, j1 = bj*blk, min((bj+1)*blk, N)
            # C_bl 에 누적
            C_bl = nmod_mat(i1-i0, j1-j0, modulus)
            for bk in range(n_blocks):
                k0, k1 = bk*blk, min((bk+1)*blk, N)

                # flint 블록 생성
                A_bl = nmod_mat(A_np[i0:i1, k0:k1].tolist(), modulus)
                B_bl = nmod_mat(B_np[k0:k1, j0:j1].tolist(), modulus)

                C_bl += A_bl * B_bl            # 모듈러 곱·합

            # 결과를 numpy 로 복사
            C_np[i0:i1, j0:j1] = _nmod_to_np(C_bl, dtype=np.int64)


    return C_np

def _matmul_mod(A_np: np.ndarray,
                B_np: np.ndarray,
                modulus: int,
                block_exp: int | None = 10,
                nth: int = 64):            # ← FLINT 내부 스레드 수

  ctx.threads = nth
  A_nm = nmod_mat(A_np.tolist(), modulus)
  B_nm = nmod_mat(B_np.tolist(), modulus)
  return _nmod_to_np(A_nm * B_nm, dtype=np.int64)

# --------------------------------------------------------------------
# ② 다중 채널 RNS 행렬곱 – 멀티스레드
# --------------------------------------------------------------------
def pp_mm_all_channels(A_rns: list[np.ndarray],
                       B_rns: list[np.ndarray],
                       q_list: list[int],
                       n_threads: int = 2,
                       block_exp: int = 10,
                       nth_per_thread: int = 32):
    """
    (A_rns[p] · B_rns[p]) mod q_list[p]  ∀p  를 동시에 계산.
    - 각 채널은 별도 스레드에서 실행 → CPU 코어 활용.
    - block_exp=10 → 블록 한 변 1024.  512 MiB 한도면 충분히 안전.
      - 필요시 더 작은 값(예: 9 → 512)으로 줄여도 됨.
    반환값: C_rns (list[np.ndarray])  –  채널별 결과
    """
    assert len(A_rns) == len(B_rns) == len(q_list)
    num_mods = len(q_list)

    # def _worker(idx):
    #     return _block_matmul_mod(A_rns[idx], B_rns[idx],
    #                              q_list[idx], block_exp)

    def _worker(idx):
      return _matmul_mod(A_rns[idx], B_rns[idx], q_list[idx], block_exp=None, nth=nth_per_thread)
      
    # 채널 수가 적으면 그냥 싱글 스레드도 OK
    if n_threads <= 1 or num_mods == 1:
        return [_worker(0)] if num_mods == 1 else [_worker(i) for i in range(num_mods)]

    C_rns = [None] * num_mods
    with ThreadPoolExecutor(max_workers=min(n_threads, num_mods)) as ex:
        futs = {ex.submit(_worker, i): i for i in range(num_mods)}
        for fut in as_completed(futs):
            C_rns[futs[fut]] = fut.result()
    return C_rns
  
from multiprocessing import Process, Queue

def _worker_mp(idx, A_np, B_np, q_mod, nth, out_q):
    # 프로세스마다 FLINT 내부 스레드 수 지정
    ctx.threads = nth
    A_nm = nmod_mat(A_np.tolist(), q_mod)
    B_nm = nmod_mat(B_np.tolist(), q_mod)
    C_nm = A_nm * B_nm
    out_q.put((idx, _nmod_to_np(C_nm, dtype=np.int64)))

def pp_mm_all_channels_mp(A_rns: list[np.ndarray],
                          B_rns: list[np.ndarray],
                          q_list: list[int],
                          nth_per_proc: int = 64):
    """
    멀티프로세싱 버전: 채널 수만큼 Process를 띄워
    각 프로세스 안에서 FLINT 내부 스레드(nth_per_proc) 병렬 수행.
    """
    num_mods = len(q_list)
    out_q    = Queue()
    procs    = []
    C_rns    = [None] * num_mods

    # 1) 각 채널별로 프로세스 시작
    for i in range(num_mods):
        p = Process(target=_worker_mp,
                    args=(i,
                          A_rns[i], B_rns[i],
                          q_list[i],
                          nth_per_proc,
                          out_q))
        p.start()
        procs.append(p)

    # 2) 결과 수집
    for _ in range(num_mods):
        idx, C = out_q.get()
        C_rns[idx] = C

    # 3) 프로세스 종료 대기
    for p in procs:
        p.join()

    return C_rns

from multiprocessing import Pool
from flint import nmod_mat, ctx
import numpy as np

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


def make_B_only_ciphertext(engine, level=0):
  """
  Return a fresh ciphertext at `level` whose decryption is a vector of 1.0:
    - A (the c1 part) is all zeros
    - B (the c0 part) is δ mod qi in every RNS channel, so
      after the usual CKKS decrypt+decode you get exactly 1.0
  """
  # scale R = 2^scale_bits
  R = 2 ** engine.ctx.scale_bits

  # choose the channel→device mapping at this level
  dest = engine.ntt.p.destination_arrays[level]
  devices = engine.ntt.devices

  B_list, A_list = [], []
  for dev_id, prime_idxs in enumerate(dest):
      dev = devices[dev_id]
      # each prime_idx corresponds to one RNS channel; 
      # all polynomials live in R[x]/(x^N+1)
      poly_len = engine.ctx.N

      # build c0 (the “B”) as a (n_ch × poly_len) tensor on the right GPU
      B_dev = torch.stack([
          torch.full((poly_len,),
                      fill_value=(((R**2) % engine.ctx.q[q_idx] )),
                      dtype=engine.ctx.torch_dtype,
                      device=dev)
          for q_idx in prime_idxs
      ], dim=0)

      # build c1 (the “A”) as zero
      A_dev = torch.zeros_like(B_dev)

      B_list.append(B_dev)
      A_list.append(A_dev)

  # wrap into a “raw” data_struct (no NTT, no Montgomery)
  return data_struct(
      data           =(B_list, A_list),
      include_special=False,
      ntt_state      =False,
      montgomery_state=False,
      origin         =presets.types.origins["ct"],
      level          =level,
      hash           =engine.hash,
      version        =engine.version
  )

def make_all_coeffs_unsigned(cts, engine):
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

def extract_B_rns(cts):
  N = len(cts)
  num_mods = cts[0].data[0][0].size(0)
  B_rns = []
  for p in range(num_mods):
    B_rns.append(np.vstack([
      cts[i].data[0][0][p].cpu().numpy() for i in range(N)
    ]))
  return B_rns



def rns_mat_to_cts(engine,
                   rns_mat: list[np.ndarray],   # 리스트 길이 = num_mods
                   template_ct: data_struct     # 레벨·origin 등 메타용
                  ) -> list[data_struct]:
    """
    RNS 행렬 리스트(채널별 (N,N)) → ciphertext 리스트 길이 N
      * 각 ciphertext 는 A=0, B=해당 행
      * 아직 NTT/몽고메리 미적용 상태 (template_ct 와 동일)
    """
    num_mods = len(rns_mat)
    N, _ = rns_mat[0].shape
    dtype   = engine.ctx.torch_dtype
    device  = torch.device('cpu')

    out_cts = []
    for i in range(N):
        # --- B 부분 구성 -------------------------------------------------
        B_chunks = [
            torch.tensor(rns_mat[p][i, :], dtype=dtype, device=device)
            for p in range(num_mods)
        ]
        # A=0
        A_chunks = [torch.zeros_like(B_chunks[0]) for _ in range(num_mods)]

        # Liberate 의 data 구조는 [device][chunk] 이므로 한 디바이스에 몰아둔다
        B_part = [torch.stack(B_chunks, dim=0)]   # shape (num_mods, poly_len)
        A_part = [torch.stack(A_chunks, dim=0)]

        mtype = -2 if template_ct.include_special else -1
        # engine.ntt.make_unsigned([B_part, template_ct.level, mtype)

        ct = data_struct(
            data            =[B_part, A_part],
            include_special =template_ct.include_special,
            ntt_state       =template_ct.ntt_state,       # False  (raw coeffs)
            montgomery_state=template_ct.montgomery_state,# False
            origin          =template_ct.origin,
            level           =template_ct.level,
            hash            =template_ct.hash,
            version         =template_ct.version
        )
        
        engine.ntt.reduce_2q(ct.data[0], template_ct.level, mtype)
        
        out_cts.append(ct)

    return out_cts


def ccmmB(engine, ctU, ctV, sk):
  
  # line 1 : transpose ctU
  # if os.path.exists("testB/ct_matrix_transpose2.pkl"):
  #   _, _, ctU_T = ct_matrix_load("testB/ct_matrix_transpose2.pkl")
  # else:
  #   ctU_T = c_mt(engine, ctU, sk = sk)
  #   ct_matrix_save(ctU_T, sk, file_name="testB/ct_matrix_transpose2.pkl") 
  
  ctU_T = c_mt(engine, ctU, sk=sk)
  
  for i in range(N):
    ctV[i]  = engine.cpu(engine.level_up(engine.cuda(ctV[i]), ctU_T[i].level))

  # for i in range(3):
  #   print(engine.decode( engine.decrypt(engine.cuda(ctV[i]), sk=sk), coeff=True))  
  # print(engine.decode(engine.decrypt(engine.cuda(ctU_T[0]), sk=sk), coeff=True))

  # print(ctU_T[0].data[0][0])
  # print(ctU_T[0].data[1][0])
  
  # line 2: M11 pp-mm
  # make ctU_T to unsigned format and print
  ctU_T[0] = engine.cpu(engine.negate(engine.cuda(ctU_T[0])))
  
  # print(engine.decode(engine.decrypt(engine.cuda(ctU_T[0]), sk=sk), coeff=True))

  
  ctU_T_us = make_all_coeffs_unsigned(ctU_T, engine)
  
  # for i in range(3):
  #     print(engine.decode(engine.decrypt(engine.cuda(ctU_T_us[i]), sk=sk), coeff=True))
  
  ctV = make_all_coeffs_unsigned(ctV, engine)
  
  # for i in range(3):
  #     print(engine.decode(engine.decrypt(engine.cuda(ctV[i]), sk=sk), coeff=True))
  
  
  #ciphertext to matrix
  B = extract_B_rns(ctU_T_us)
  Bp = extract_B_rns(ctV)

  # flint library multiplication
  M11 = pp_mm_all_channels_mp_fast(B, Bp, engine.ctx.q[:len(B)], 64)

  out = rns_mat_to_cts(engine, M11, ctV[0]) 
  # line 7 return and decrypt
  for i in range(3):
    rs = engine.decode(engine.decrypt(out[i], sk), coeff=True)
    print(rs[:10])
    
  return out
    
  

if __name__ == "__main__":
    # from liberate import fhe
    # from liberate.fhe import presets
    # # 1) build engine & keys
  
    # engine = fhe.ckks_engine(logN =13, buffer_bit_length = 62, scale_bits=40, num_special_primes=1, verbose=True)
    # sk  = engine.create_secret_key()
    # pk = engine.create_public_key(sk)

    # # 2) make your “all-1” B-only ciphertext
    # N = engine.ctx.N
    # cts = []
    # for i in range(N):
    #   cts.append(engine.cpu(make_B_only_ciphertext(engine)))
    
    # ccmmB(engine, cts, cts, sk)
    
    
    # print("----------test-------------")
    # from liberate import fhe
    # from liberate.fhe import presets
    # engine = fhe.ckks_engine(logN= 13, buffer_bit_length = 62, scale_bits = 40, num_special_primes=1, verbose=True)
    
    # sk = engine.create_secret_key()
    # pk = engine.create_public_key(sk)
    # N = engine.ctx.N
    
    # # make matrix 
    # def make_ctx(engine, pk, N):
    #   poly = np.ones(N)
    #   encrypt_poly = engine.encrypt(engine.encode(poly, coeff=True), pk)
    #   print("encrypted_poly form:", encrypt_poly)
    #   for dev_chunk_list in encrypt_poly.data[1][0]:
    #     for chunk in dev_chunk_list:
    #       chunk.zero_()
    #   print()
    #   return encrypt_poly    
    
    # cx = []
    # ptx = []
    
    # for i in range(N):
    #   cx.append(engine.cpu(make_ctx(engine, pk, N)))
    #   ptx.append(engine.decode(engine.decrypt(cx[i], sk), coeff=True))
    #   if i == 0:
    #     print("ptx form:",ptx[0])
    
    # out = ccmmB(engine, cx, cx, sk)
    
    
    

  import numpy as np
  from pprint import pprint
  import time

  # ---------------------- 실험 파라미터 --------------------------
  N      = 2 ** 14 # 행렬 크기 (작게 두고 빠르게 검증)
  q_list = [2147483647, 2147483647,2147483647]  # RNS 1채널: 31-bit 소수 하나만 사용
  VAL    = (1 << 60)-1      # 모든 원소 값 (unsigned 1)
  print("val:", VAL)
  # --------------------------------------------------------------

  # 1. 테스트용 행렬 세트 만들기  (길이 L 리스트, 여기선 L=1)
  A_rns  = [np.full((N, N), VAL, dtype=np.int64), np.full((N,N), VAL, dtype=np.int64),np.full((N,N), VAL, dtype=np.int64),np.full((N,N), VAL, dtype=np.int64)]
  B_rns  = [np.full((N, N), VAL, dtype=np.int64), np.full((N,N), VAL, dtype=np.int64),np.full((N,N), VAL, dtype=np.int64),np.full((N,N), VAL, dtype=np.int64)]

  # 2. pp_mm_all_channels (당신 코드) 호출
  t0 = time.perf_counter()
  C_rns = pp_mm_all_channels_mp_fast(A_rns, B_rns, q_list, 64)
  elapsed = time.perf_counter() - t0
  print(f"pp_mm_all_channels 실행 시간: {elapsed:.2f} 초")

  C     = C_rns[0]          # 단일 채널 결과

  # 3. 파이썬 무한정밀로 레퍼런스 계산
  C_ref = (A_rns[0].astype(object) @ B_rns[0].astype(object)) % q_list[0]
  C_ref = C_ref.astype(np.int64)

  # 4. 비교
  if np.array_equal(C, C_ref):
      print("✅  PASS  –  pp_mm_all_channels is bit-exact for all-ones input.")
  else:
      print("❌  FAIL  –  mismatch detected!")
      # 첫 불일치 위치와 값을 보여줌
      i, j = np.argwhere(C != C_ref)[0]
      print(f"  at ({i},{j}) → {C[i,j]}   ref={C_ref[i,j]}")

  # (옵션) 결과 일부 프린트
  print("\nC[0:4,0:4] =")
  print(C[:4, :4])
  

