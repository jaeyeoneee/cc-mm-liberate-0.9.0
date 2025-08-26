import numpy as np
import torch
import math

from liberate import fhe
from liberate.fhe import presets
from liberate.fhe.data_struct import data_struct

class CCMM:
  
  def __init__(self):
    # engine 
    self.params = presets.params["bronze"]
    self.engine = fhe.ckks_engine(**self.params)
    
    # key
    self.sk = self.engine.create_secret_key()
    self.pk = self.engine.create_public_key(self.sk)

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

  def normalize_unsinged(self, chunk: torch.Tensor, level, include_special):
    mult_type = -2 if include_special else -1
    self.engine.ntt.make_unsigned([chunk], level, mult_type)
    self.engine.ntt.reduce_2q(    [chunk], level, mult_type)
    return chunk

  def rotate_with_cyclic_sign(self, ct, shift):
    """
    multiply ct(X) polynomial by x^i
    """
    
    r = shift // self.slot_size
    s = shift % self.slot_size
    
    ct = self.engine.cuda(ct)
    shifted_data = []
    
    for comp in ct.data:
      shifted_comp = []
      for chunk in comp:
        rolled = torch.roll(chunk, shifts=shift, dims=-1)
        if s != 0:
          rolled[..., :s] *= -1
        if (r % 2) == 1:
          rolled *= -1
        self.normalize_unsinged(chunk, ct.level, ct.include_special)
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
        
        
  def tweak(self):
    pass
  
  def cmt(self):
    pass
  
  def ccmm(self):
    pass