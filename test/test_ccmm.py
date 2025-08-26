from ccmm.ccmm import CCMM
import logging
import numpy as np

def test_init():
  
  model = CCMM()

def test_engine_info():
  
  model = CCMM()

  model.engine_info()

def test_encode():
  
  model = CCMM()
  
  a = np.ones((model.slot_size, model.slot_size))
  
  model.encode(a)
  
def test_decode():
  
  model = CCMM()
  
  a = np.zeros((model.slot_size, model.slot_size), dtype=np.float64)
  a[1::2, :] = 1.0
  
  cts = model.encode(a)
  
  matrix = model.decode(cts)
  
  assert np.allclose(a, matrix, rtol = 1e-5)


def test_rotate_with_cyclic_sign_np():
  
  model = CCMM()
  
  vector = np.arange(5)
  
  print(vector)

  shift1 = 7
  shift2 = 12
  
  vector1 = model.rotate_with_cyclic_sign_np(vector, shift1)
  vector2 = model.rotate_with_cyclic_sign_np(vector, shift2)
  
  assert np.allclose(vector1, np.array([3, 4, 0, -1, -2]), rtol=1e-5)
  assert np.allclose(vector2, np.array([-3, -4, 0, 1, 2]), rtol=1e-5)

def test_tweak_np():
  
  model = CCMM()
  
  matrix = np.arange(16).reshape(4, 4)
  
  print()
  print("----------test-tweak-np--------------")
  print(f"original matrix: {matrix}")
  
  n, N = matrix.shape
  
  tweaked = []
  for j in range(n):
    tweaked_row = np.zeros((N,))
    for i in range(n):
      shift = 2*i*j*(N//n)
      tweaked_row += model.rotate_with_cyclic_sign_np(matrix[i], shift)
    tweaked.append(tweaked_row)
  expected = np.vstack(tweaked)
  print(f"expected tweak result: {expected}")
  
  tweak_result = model.tweak_np(matrix)
  print(f"tweak result:{tweak_result}")
  
  assert np.allclose(expected, tweak_result, rtol=0)
  
def test_rotate_with_cyclic_sign():
  """cuda, cpu 조심해서 할당하기!
  """
  model = CCMM()
  
  shift =2
  
  vector = np.arange(2**14)//10
  encoded = model.engine.encode(vector, coeff=True)
  encrypted = model.engine.encrypt(encoded, model.pk)
  encrypted = model.engine.cpu(encrypted)
  
  rotated = model.rotate_with_cyclic_sign(encrypted, shift)
  rotated = model.engine.cuda(rotated)
  decrypted = model.engine.decrypt(rotated, model.sk)
  decoded = model.engine.decode(decrypted, coeff=True)
  
  expected = model.rotate_with_cyclic_sign_np(vector, shift)
  
  assert np.allclose(decoded, expected, rtol=1e-5)
  