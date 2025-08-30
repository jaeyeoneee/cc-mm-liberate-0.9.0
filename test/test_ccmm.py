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
  
  # matrix = np.arange(16).reshape(4, 4)
  matrix = [[0, 1, 2, 3], [ -7, 4, 5, 6], [-10, -11, 8, 9], [-13, -14, -15, 12]]
  matrix = np.array(matrix)
  
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

def test_tweak():
  
  model = CCMM()
  
  N = model.slot_size
  matrix = np.random.rand(N, N) * 10
  
  tweak_np = model.tweak_np(matrix)
  
  encrypted = model.encode(matrix)
  
  tweak_liberate = model.tweak(encrypted)
  
  decrypted = model.decode(tweak_liberate)
  
  assert np.allclose(tweak_np, decrypted, rtol=1000)

  
def test_encode_overflow():
  
  model = CCMM()
  
  input = np.full((model.slot_size,), fill_value = 100000000000)
  
  encoded = model.engine.encode(input, coeff=True)
  encrypted = model.engine.encrypt(encoded, model.pk)
  
  print(f"encrypted : {encrypted}")
  
  decrypted = model.engine.decrypt(encrypted, model.sk)
  decoded = model.engine.decode(decrypted, coeff=True)
  
  print(decoded)
  
def test_automorphism_np():
  
  model = CCMM()
  
  input = np.array([1, 2, 3, 4])
  shift = 3
  
  expected_result = np.array([1, 4, -3, 2])
  
  result = model.automorphism_np(input, shift)
  
  assert np.allclose(expected_result, result)
  

def test_cmt_np():
  
  model = CCMM()
  
  input = np.arange(16).reshape(4, 4)
  
  result = model.cmt_np(input)
  
  assert np.allclose(input.T, result)
  
def test_cmt():
  
  model = CCMM()
  
  N = model.slot_size
  matrix = np.random.rand(N, N) * 10
  
  encrypted = model.encode(matrix)
  
  cmt = model.cmt(encrypted)
  
  decrypted = model.decode(cmt)
  
  # decrypted = model.cmt_np(matrix)
  
  print(f"real matrix.T: {matrix.T}")
  print(f"transpose matrix T: {decrypted}")
  
  assert np.allclose(matrix.T, decrypted, rtol=1000)
  
def test_cmt_twice():
  
  model = CCMM()
  
  N = model.slot_size
  matrix = np.random.rand(N, N) * 10
  
  encrypted = model.encode(matrix)
  
  cmt = model.cmt(encrypted)
  cmt_twice = model.cmt(cmt)
  
  decrypted = model.decode(cmt_twice)
  
  print(f"real matrix: {matrix}")
  print(f"computed matrix: {decrypted}")
  
  assert np.allclose(matrix, decrypted, rtol = 1000)

def test_decrypt_scale():
  from liberate import fhe
  from liberate.fhe import presets
  from liberate.fhe.data_struct import data_struct
  
  
  engine1 = fhe.ckks_engine(logN= 13, buffer_bit_length = 62, scale_bits = 20, num_special_primes=1, verbose=True)
  params = presets.params["bronze"]
  engine2 = fhe.ckks_engine(**params)
  
  sk1 = engine1.create_secret_key()
  sk2 = engine2.create_secret_key()
  
  pk1 = engine1.create_public_key(sk1)
  pk2 = engine2.create_public_key(sk2)
  
  input1 = np.arange(engine1.ctx.N) % 10
  input2 = np.ones((engine2.ctx.N, ))
  
  encoded1 = engine1.encode(input1, coeff=True)
  encoded2 = engine2.encode(input2, coeff=True)
  
  encrypted1 = engine1.encrypt(encoded1, pk1)
  encrypted2 = engine2.encrypt(encoded2, pk2)
  
  decrypted1 = engine1.decrypt(encrypted1, sk1)
  decrypted2 = engine2.decrypt(encrypted2, sk2)
  
  decoded1 = engine1.decode(decrypted1, coeff=True)
  decoded2 = engine2.decode(decrypted2, coeff=True)
  
  print(f"decoded1: {decoded1}")
  print(f"decoded2: {decoded2}")
  

def test_level1_matrix():
  
  model = CCMM()
  
  max_level = model.engine.num_levels - 2
  N = model.slot_size
  matrix = np.random.rand(N, N) * 10
  
  cts = model.encode(matrix)

  new_cts = model.make_level1_matrix(cts)
  
  for ct in new_cts:
    assert ct.level == max_level

def test_extract_AB_rns():
  
  model = CCMM()
  
  N = model.slot_size
  matrix = np.random.rand(N, N) * 10
  
  encrypted = model.encode(matrix)
  encrypted_level1 = model.make_level1_matrix(encrypted)

  A, B = model.extract_matrices_from_cts(encrypted_level1)

  builed = model.build_cts(A, B, encrypted_level1[0])
  
  decrypted_original = model.decode(encrypted)
  decrypted_level1 = model.decode(encrypted_level1)
  decrypted_builed = model.decode(builed)
  
  print(f"original matrix:{matrix}")
  
  print(f"decrypted original:{decrypted_original}")
  print(f"decrypted_level1:{decrypted_level1}")
  print(f"decrypted_builed:{decrypted_builed}")
  
  assert np.allclose(matrix, decrypted_builed, rtol=10)  
  
def test_modular_q():
  
  model = CCMM()
  
  model.engine_info()
  
  pt = np.ones((model.slot_size, ))
  ct = model.engine.encode(pt, coeff=True)
  ct = model.engine.encrypt(ct, model.pk)
  
  ct = model.engine.level_up(ct, 1)
  
  dest = model.engine.ntt.p.destination_arrays[ct.level][0]

  print(f"dest: {dest}")
  print(f"q: {model.q[dest[0]:dest[-1]+1]}")
  print(f"first modular list q size: {ct.data[0][0].size(0)}")


def test_bonly_ppmm_rescale_to_N():
    """
    B만 사용해 Algorithm 3의 PP-MM(+rescale) 경로를 검증한다.
    - B1 = Δ * 1_{N×N}, B2 = Δ * 1_{N×N}
    - PP-MM: out = (B1 @ B2) mod q_i  = N * Δ^2 (mod q_i)
    - 'rescale' 모사: out_scaled = out * Δ^{-1} (mod q_i)  =>  N * Δ (mod q_i)
    - ciphertext로 구성 후 decode(coeff=True) ≈ N 이어야 한다.
    """

    model = CCMM()
    eng = model.engine
    N = model.slot_size
    scale = eng.scale

    dummy_row = np.zeros(N, dtype=np.float64)
    pt = eng.encode(dummy_row, coeff=True)
    ct_template = eng.cpu(eng.encrypt(pt, model.pk))
    # 레벨(활성 RNS 채널 범위) 획득
    level = ct_template.level

    dest = eng.ntt.p.destination_arrays[level][0]
    q_list = [int(q) for q in model.q[dest[0]:dest[-1]+1]]
    L = len(q_list)  # 사용되는 모듈러 채널 수

    B1_rns = [np.full((N, N), (scale**2) % qi , dtype=np.int64) for qi in q_list]
    B2_rns = [np.full((N, N), (scale**2) % qi , dtype=np.int64) for qi in q_list]

    ctB1 = model.build_cts(None, B1_rns, ct_template)
    ctB2 = model.build_cts(None, B2_rns, ct_template)
    
    matB1 = model.decode(ctB1)
    matB2 = model.decode(ctB2)
    assert np.allclose(matB1, np.ones((N, N), dtype=np.float64), atol=5e-3)
    assert np.allclose(matB2, np.ones((N, N), dtype=np.float64), atol=5e-3)

    ctB1_T = model.cmt(ctB1)
    _, B1_rns_T = model.extract_matrices_from_cts(ctB1_T)

    out_rns = model.blas_mat_mult(B1_rns_T, B2_rns, level)

    invΔ = [pow(scale % qi, -1, qi) for qi in q_list]
    out_rns_scaled = [
        (out_rns[i].astype(np.int64) * invΔ[i]) % q_list[i] for i in range(L)
    ]

    ct_out = model.build_cts(None, out_rns_scaled, ct_template)

    mat_out = model.decode(ct_out)
    assert np.allclose(mat_out, np.full((N, N), float(N)), atol=5e-2)

    print("Decoded(out)[0,:10] =", mat_out[0, :10])

def test_rescale():
  
  model = CCMM()
  
  N = model.slot_size
  scale = model.scale
  pt = np.full((N, ), scale)
  
  encoded = model.engine.encode(pt, coeff=True)
  encrypted = model.engine.encrypt(encoded, model.pk)
  
  rescaled = model.engine.rescale(encrypted)
  
  decrypted = model.engine.decrypt(encrypted, model.sk)
  decoded = model.engine.decode(decrypted, coeff=True)
  
  rescaled_decrypted = model.engine.decrypt(rescaled, model.sk)
  rescaled_decoded = model.engine.decode(rescaled_decrypted, coeff=True)
  
  print(f"original: {decoded}")
  print(f"rescaled: {rescaled_decoded}")
  
  expected = np.zeros((N, ))
  
  assert np.allclose(expected, rescaled_decoded, rtol = 1)

def test_rescale_matrix():
  
  model = CCMM()
  
  N = model.slot_size
  scale = model.scale
  pt = np.full((N, N), scale)
  
  encrypted = model.encode(pt)
  
  rescaled = model.rescale_matrix(encrypted)
  
  decrypted = model.decode(rescaled)
  
  expected = np.zeros((N, ))
  for ct in decrypted:
    assert np.allclose(expected, ct, rtol=1)

def test_relinearize_matrix():
  
  model = CCMM()
  
  N = model.slot_size
  matrix = np.random.rand(N, N) * 10
  
  encrypted = model.encode(matrix)
  encrypted_level1 = model.make_level1_matrix(encrypted)
  
  A, B = model.extract_matrices_from_cts(encrypted_level1)
  triple_ct = model.build_triple_cts_relin(A, B, None, encrypted_level1[0])
  
  print(f"relin result: {triple_ct[0]}")
  
def test_coeff_add():
  
  model = CCMM()
  N = model.slot_size
  
  pt1 = np.full((N, ), 2)
  pt2 = np.full((N, ), 5)
  
  encoded1 = model.engine.encode(pt1, coeff=True)
  encoded2 = model.engine.encode(pt2, coeff=True)
  
  encrypted1 = model.engine.encrypt(encoded1, model.pk)
  encrypted2 = model.engine.encrypt(encoded2, model.pk)
  
  added = model.engine.add(encrypted1, encrypted2)
  
  decrypted = model.engine.decrypt(added, model.sk)
  decoded = model.engine.decode(decrypted, coeff=True)
  print(f"encrypted1: {encrypted1}")
  print(f"encrypted2: {encrypted2}")
  print(f"added: {added}")
  print(f"decoded : {decoded}")
  
  assert np.allclose(decoded, pt1+pt2, rtol=1)

def test_add_matrix():
  
  model = CCMM()
  
  N = model.slot_size
  matrix1 = np.random.rand(N, N) * 10
  matrix2 = np.random.rand(N, N) * 10
  
  encrypted1 = model.encode(matrix1)
  encrypted2 = model.encode(matrix2)
  
  added = model.add_matrix(encrypted1, encrypted2)
  
  decrypted = model.decode(added)
  
  assert np.allclose(decrypted, matrix1 + matrix2, rtol = 1)
  

def test_ccmm():
  
  model = CCMM()
  
  N = model.slot_size
  matrix1 = np.random.rand(N, N) * 10
  matrix2 = np.random.rand(N, N) * 10
  
  ctu = model.encode(matrix1)
  ctv = model.encode(matrix2)
  
  result = model.ccmm(ctu, ctv)
  
  decrypted_result = model.decode(result)
  
  expected_result = matrix1 @ matrix2
  
  print(f"decrypted_result: {decrypted_result}")
  print(f"expected_result: {expected_result}")
  
  assert np.allclose(decrypted_result, expected_result, rtol=100)
