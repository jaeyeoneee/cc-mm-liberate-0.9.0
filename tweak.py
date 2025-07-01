import numpy as np
import torch
import math

from liberate import fhe
from liberate.fhe import presets
from liberate.fhe.data_struct import data_struct

np.random.seed(0)

def cosine_basis_input(n: int, N: int) -> list[np.ndarray]:
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return [np.cos((i + 1) * x) for i in range(n)]

def dft_basis_input(n: int, N: int) -> list[np.ndarray]:
    basis = []
    for k in range(n):
        vec = np.zeros(N)
        for j in range(N):
            vec[j] = np.cos(2 * np.pi * k * j / N)  
        basis.append(vec)
    return basis


def random_input(n:int, N:int)->list[np.ndarray]:
    return np.random.rand(n, N) *10

def rotate_with_cyclic_sign(vec: np.ndarray, k: int) -> np.ndarray:
    N = vec.shape[-1]
    # 몫(r)과 나머지(s)
    r = k // N
    s = k % N

    out = np.roll(vec, s)

    if s > 0:
        out[:s] *= -1

    if (r % 2) == 1:
        out *= -1

    return out

def polynomial_X_mult(engine, ct, i):
    """
    multiply ct(X) polynomial by X^i
    Require: A ciphertext ct(X) and an integer i
    Ensure: A ciphertext ct'(X) such that ct(X)' = X^i * ct(X) (ct(X) = ct_0(X) + ct_1(X) * X) 
    작동 잘함! data를 shift 하면 원래 m(x)에서 shift 가 일어나는 것 확인. 
    X^N 이상이 곱해지면 - 가 붙는 거 주의해야 함..
    CPU에 있는 벡터를 받아, CUDA로 옮겨서 연산을 수행하고, 
    다시 CPU로 옮겨서 반환한다.
    """
    # print("i:", i)
    # print(ct)
    # print("ct data 0]:",ct.data[0])
    
    ct = engine.cuda(ct)
    
    shifted_data = []
    
    def normalize_unsigned(chunk: torch.Tensor, level, include_special):
        mult_type = -2 if include_special else -1
        engine.ntt.make_unsigned([chunk], level, mult_type)
        engine.ntt.reduce_2q(    [chunk], level, mult_type)
        return chunk
    
    for comp in ct.data:
        shifted_comp = []
        for chunk in comp:
            
            N = chunk.shape[-1]
            # print(N)
            r = i // N
            s = i % N
            
            rolled = torch.roll(chunk, shifts=i, dims=-1)
            
            if s != 0:
                rolled[..., :s] *= -1  # cyclic 구조!!!
            
            if (r % 2) == 1:
                rolled *= -1
            
            rolled = normalize_unsigned(rolled, ct.level, ct.include_special)
            
            shifted_comp.append(rolled)
        shifted_data.append(shifted_comp)
    
    return engine.cpu(data_struct(
        data=shifted_data,
        include_special=ct.include_special,
        ntt_state=ct.ntt_state,
        montgomery_state=ct.montgomery_state,
        origin=ct.origin,
        level=ct.level,
        hash=ct.hash,
        version=ct.version
    ))
    
    
def tweak(engine, cts, matrix = None, sk=None):
    """
    Algorithm 1: TWEAK
    Require: A power-of-two integer n, and n ciphertexts ct
    Ensure: Ciphertexts ct' such that ct_j' = sum_i(X^2ijN/n ct_i) for each j in [0, n-1]
    input 으로 cpu에 있는 행렬을 리스트로 받아오고, 
    연산 과정에서 필요할 때마다 gpu에서 cuda로 옮겨서 연산한다.
    이후에 반환은 cpu에 있는 tweak 결과값을 반환한다.
    """
    
    n = len(cts)
    # print(n)
    
    if n == 1:
        # return cts, matrix.copy()
        return cts
    
    ct_p = [None] * n # 결과 저장
    # result_blocks = [None] * n
    
    ct_p[0] = cts[0] #cpu
    # result_blocks[0] = matrix[0].copy()
    
    for l in range(0, int(math.log(n, 2))):
        pow2 = 2**l
        block = n // (2 * pow2)

        aux_cts = [cts[(2*j+1)*block] for j in range(pow2)] #cpu
        # aux_inputs = [matrix[(2*j + 1) * block] for j in range(pow2)]        
        # print("aux_cts:", len(aux_cts))
        # aux, aux_np = tweak(engine, aux_cts, aux_inputs, sk) #cpu
        aux = tweak(engine, aux_cts)

        for j in range(0, pow2):
            shift_k = (engine.ctx.N // (2**l)) * j

            ct_rot = engine.cuda(polynomial_X_mult(engine, aux[j], shift_k)) #cpu
            # print("ct_rot rs:", engine.decode(engine.decrypt(ct_rot, sk), coeff=True)[:100])
            # rotated = rotate_with_cyclic_sign(aux_np[j], shift_k)
            # print("rotated np rs:", rotated[:100])

            temp = engine.cuda(ct_p[j])
            
            # while temp.level > ct_rot.level:
            #     temp = engine.rescale(temp)
            # while ct_rot.level > temp.level:
            #     ct_rot = engine.rescale(ct_rot)
            # print(temp.level, ct_rot.level)
            
            ct_sub = engine.sub(temp, ct_rot)
            ct_add = engine.add(temp, ct_rot)
            # print("temp_decrypt:", engine.decrypt(temp, sk)[0])
            # print("ct_rot_decrypt:", engine.decrypt(ct_rot, sk)[0])
            # print("temp:", engine.decode(engine.decrypt(temp, sk), coeff=True)[:100])
            # print("ct_rot:", engine.decode(engine.decrypt(ct_rot, sk),coeff=True)[:100])
            
            # if l==9:
            #     ct_add = engine.rescale(ct_add)
            #     ct_sub = engine.rescale(ct_sub)
            
            ct_p[j+pow2] = engine.cpu(ct_sub)
            ct_p[j] = engine.cpu(ct_add)
            # result_blocks[j+pow2] = result_blocks[j] - rotated
            # result_blocks[j]      = result_blocks[j] + rotated
            
            # print("ct_p j+pow2 rs:", engine.decode(engine.decrypt(engine.cuda(ct_p[j+pow2]), sk), coeff=True)[:100])
            # print("ct_p j rs:", engine.decode(engine.decrypt(engine.cuda(ct_p[j]), sk), coeff=True)[:100])

            # print("ct_np j+pow2 rs:", result_blocks[j+pow2][:100])
            # print("ct_np j rs:", result_blocks[j][:100])
            # print("ct_p j decode:", engine.decrypt(engine.cuda(ct_p[j]), sk))

        
    return ct_p
    
def multiply_by_X_test():
    params = presets.params["bronze"]
    engine = fhe.ckks_engine(**params)
    
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    
    N = engine.ctx.N
    rng = np.random.default_rng(0)
    slot = rng.integers(0, 100, size=N//2)
    ct = engine.encode(slot)
    print("ct:",ct)
    ct = engine.encrypt(ct, pk)
    ct = engine.cpu(ct)
    
    ct_p = polynomial_X_mult(engine, ct, 1)
    decrypted = engine.decrypt(engine.cuda(ct_p), sk)
    print("decrypted:", decrypted)
    

def tweak_np(matrix: np.ndarray) -> np.ndarray:
    """
    Algorithm 1 TWEAK의 numpy 버전.
    Require:
      - matrix.shape == (n, N), n은 2^k
    Ensure:
      - result[j] = sum_{i=0..n-1} X^{2·i·j·(N/n)} * matrix[i]
        (여기서 X^k 곱셈은 rotate_with_cyclic_sign 함수로 구현)
    """
    n, N = matrix.shape
    # 기저: n=1이면 그대로 반환
    if n == 1:
        return matrix.copy()

    # 각 레벨의 결과를 담을 리스트
    result_blocks = [None] * n

    result_blocks[0] = matrix[0].copy()
    
    # log₂(n) 레벨 반복
    levels = int(np.log2(n))
    for ell in range(levels):
        size = 2**ell                     # 2^ℓ
        block = n // (2 * size)           # n / 2^{ℓ+1}

        # (2j+1)블록만 뽑아서 재귀 호출
        aux_inputs = [matrix[(2*j + 1) * block] for j in range(size)]
        # 각 aux_inputs는 길이 N의 배열이니, (size, N) 형태로 쌓아서 재귀
        aux = tweak_np(np.vstack(aux_inputs))  # 반환도 shape (size, N)

        # 이 레벨에서 두 갈래로 나눠 합·차 계산
        for j in range(size):
            # shift = (N / 2^ℓ) * j
            shift_k = (N // (2**ell)) * j
            rotated = rotate_with_cyclic_sign(aux[j], shift_k)

            # 상위 레벨부터 덧셈·뺄셈을 original matrix 값으로
            result_blocks[j+size] = result_blocks[j] - rotated
            result_blocks[j]      = result_blocks[j] + rotated

    # 리스트 다시 (n, N) 배열로
    return np.vstack(result_blocks)


def tweak_test(n=2**14):
    """
    TODO: n이 N과 동일해야 하는지 확인할 필요가 있음. 2의 거듭제곱이면 만족하는 듯!
    """
    params = presets.params["bronze"]
    engine = fhe.ckks_engine(**params)
    
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    
    # 임의의 정수 다항식 n개 생성, 암호화한다.
    N = engine.ctx.N
    rng = np.random.default_rng(0)
    slots = [rng.integers(0, 100, size = N//2) for _ in range(n)]
    pts = [engine.encode(s) for s in slots]
    pts_np = [pt[0].cpu().numpy() for pt in pts]
    cts = [engine.cpu(engine.encrypt(pt, pk)) for pt in pts]    

    # tweak 적용
    tweaked_cts = tweak(engine, cts)
    # n의 크기가 크지 않기 때문에 여기에서는 cuda에 옮겨서 계산한다.
    # tweaked_cts = [engine.decrypt(engine.cuda(ct), sk) for ct in tweaked_cts]
    
    # tweak nuumpy
    tweaked_cts_np = tweak_np(np.array(pts_np))
    
    # # 복호 결과와 수학적 기대치 비교
    # expected = []
    # for j in range(n):
    #     acc = np.zeros_like(pts[0][0].cpu().numpy())
    #     for i in range(n):
    #         k = 2*i*j*(N//n)
    #         acc += rotate_with_cyclic_sign(pts[i][0].cpu().numpy(), k)
    #     expected.append(acc)
        
    for j in range(n):
        rs = engine.decrypt(engine.cuda(tweaked_cts[j]), sk)[0][0].cpu().numpy()
        print("tweaked:", rs)
        # print("expected:", expected[j])
        print("tweak np:", tweaked_cts_np[j])
    

def tweak_test_encode(n=2**8):
    """
    TODO: n이 N과 동일해야 하는지 확인할 필요가 있음. 2의 거듭제곱이면 만족하는 듯!
    """
    params = presets.params["bronze"]
    engine = fhe.ckks_engine(**params)
    
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    
    # 임의의 정수 다항식 n개 생성, 암호화한다.
    N = engine.ctx.N
    q = engine.ctx.q[0] #최상단 모듈러스를 사용해준다.
    
    # pts_np = [np.arange(N) % 10 for _ in range(n)]
    pts_np = random_input(n, N)
    print("pts_np:", pts_np)
    pts = [engine.encode(m, coeff=True) for m in pts_np]
    print("pts:", engine.decode(pts[0], coeff=True))
    cts = [engine.cpu(engine.encrypt(pt, pk)) for pt in pts]    

    # tweak 적용
    tweaked_cts, tweaked_cts_np = tweak(engine, cts, pts_np, sk)
    # n의 크기가 크지 않기 때문에 여기에서는 cuda에 옮겨서 계산한다.
    tweaked_cts = [engine.decrypt(engine.cuda(ct), sk) for ct in tweaked_cts]
    
    # tweak nuumpy
    # tweaked_cts_np = tweak_np(np.array(pts_np))
    
    # def centered_mod_q(vec, q):
    #     vec = np.array(vec)
    #     vec = np.where(vec > q/2, vec - q, vec)
    #     return vec
    
    print(tweaked_cts)
    for j in range(20):
        rs = engine.decode(tweaked_cts[j], coeff=True)
        print("tweaked:", rs[:100])
        # print("expected:", expected[j])
        print("tweak np:", tweaked_cts_np[j][:100])

if __name__ == "__main__":
    # multiply_by_X_test()
    tweak_test(2**8)
    
    tweak_test_encode(2**12)
    
    # print(sinusoidal_input(10 ,10))
    # roll and add decrypt test
    
    # params = presets.params["bronze"]
    # engine = fhe.ckks_engine(**params)
    
    # sk = engine.create_secret_key()
    # pk = engine.create_public_key(sk)
    
    # N = 2**14
    
    # m = np.arange(engine.ctx.N)
    
    # pts = engine.encode(m, coeff=True)
    # print("pts:", pts[0])
    # ct = engine.encrypt(pts, pk)
    # ct = engine.rescale(ct)
    # print("rescale:", engine.decrypt(ct, sk)[0])
    # m = engine.decode(pts, coeff = True)
    # pts1 = [torch.arange(0,N, dtype=torch.int64, device="cuda:0")]
    # pts2 = [torch.arange(0,N, dtype=torch.int64, device="cuda:0")]
    
    # cts1 = engine.encrypt(pts1, pk)
    # cts2 = engine.encrypt(pts2, pk)
    
    # add = engine.add(cts1, cts2)
    
    # roll1 = polynomial_X_mult(engine, engine.cpu(cts1), 1)
    # roll2 = polynomial_X_mult(engine, engine.cpu(cts2), 1)
    
    # add_roll = engine.add(engine.cuda(roll1), engine.cuda(roll2))
    
    # print(engine.decrypt(add, sk)[0].cpu().numpy())
    # print(engine.decrypt(add_roll, sk)[0].cpu().numpy())
    
    
    
    # n = 2 ** 9
    # mat = np.arange(n*n).reshape(n ,n)
    # print('input:\n', mat)
    # out = tweak_np(mat)
    # print('tweak_np output:\n', out)

    # n, N = mat.shape
    # expected = np.vstack([
    #     sum(rotate_with_cyclic_sign(mat[i], 2*i*j*(N//n)) for i in range(n))
    #     for j in range(n)
    # ])
    # print('expected:\n', expected)
    

