import functools
import math
import random
import pytest
import uuid
import numpy as np

from secrets import token_bytes

from fedbiomed.common.secagg._lom import PRF, LOM  # Assuming the class definitions are in lom.py
from fedbiomed.common.secagg import SecaggLomCrypter
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture(scope="module")
def pairwise_keys():


    nonce = token_bytes(16)

    node_ids = ['node-1', 'node-2', 'node-3']

    p_secrets_1 = {'node-2': b'\x02' * 32, 'node-3': b'\x02' * 32}
    p_secrets_2 = {'node-1': b'\x02' * 32, 'node-3': b'\x02' * 32}
    p_secrets_3 = {'node-1': b'\x02' * 32, 'node-2': b'\x02' * 32}


    return (node_ids, nonce, (p_secrets_1, p_secrets_2, p_secrets_3 ))


def test_01_lom_module_prf():
    nonce = b'\x00' * 16
    prf = PRF(nonce)

    pairwise_secret = b'\x01' * 32
    round = 1
    key = prf.eval_key(pairwise_secret, round)
    assert len(key) == 32

    seed = key
    input_size = 10
    buffer = prf.eval_vector(seed, round, input_size)
    vector = np.frombuffer(buffer, dtype='uint64')
    assert len(vector) == input_size


    # Test with big input size
    input_size = 100000000
    buffer = prf.eval_vector(seed, round, input_size)
    vector = np.frombuffer(buffer, dtype='uint64')
    assert len(vector) == input_size

def test_02_lom_protect_and_aggregate(pairwise_keys):

    node_ids, nonce, pwkeys = pairwise_keys
    lom_1 = LOM(nonce=nonce)
    lom_2 = LOM(nonce=nonce)
    lom_3 = LOM(nonce=nonce)


    tau = 1
    x_1_tau = [11111, 21111, 311111, 41111, 51111, 11116]
    protected_vector_1 = lom_1.protect(node_ids[0], pwkeys[0], tau, x_1_tau, node_ids)
    assert len(protected_vector_1) == len(x_1_tau)

    x_2_tau = [23131231,1231232,2342343, 32434, 2432345, 2343246]
    protected_vector_2 = lom_2.protect(node_ids[1], pwkeys[1], tau, x_2_tau, node_ids)
    assert len(protected_vector_2) == len(x_2_tau)

    x_3_tau = [2343241, 2342342, 4443, 34444, 2225, 2342346]
    protected_vector_3 = lom_3.protect(node_ids[2], pwkeys[2], tau, x_3_tau, node_ids)
    assert len(protected_vector_3) == len(x_3_tau)

    list_y_u_tau = [protected_vector_1, protected_vector_2, protected_vector_3]
    aggregated_vector = lom_1.aggregate(list_y_u_tau)
    assert len(aggregated_vector) == len(x_1_tau)
    sum_x = np.sum(np.array([x_1_tau, x_2_tau, x_3_tau]), axis=0)
    assert aggregated_vector == sum_x.tolist()


def test_02_lom_protect_error_case_int_to_big_to_convert(pairwise_keys):
    """This function tests the case there int is too big to convert"""

    node_ids, nonce, pwkeys = pairwise_keys
    lom_1 = LOM(nonce)

    params = [112341234, 123151234]
    lom_1.protect(node_ids[0], pwkeys[0], 1, params, node_ids)
    exit()


def test_03_lom_protect_big_int(pairwise_keys):

    node_ids, nonce, pwkeys = pairwise_keys
    lom_1 = LOM(nonce=nonce)
    lom_2 = LOM(nonce=nonce)
    lom_3 = LOM(nonce=nonce)

    tau = 1

    r_int = random.getrandbits(26)
    params = [r_int, r_int]
    protected_vector_1 = lom_1.protect(node_ids[0], pwkeys[0], tau, params, node_ids)
    protected_vector_2 = lom_2.protect(node_ids[1], pwkeys[1], tau, params, node_ids)
    protected_vector_3 = lom_3.protect(node_ids[2], pwkeys[2], tau, params, node_ids)

    pvectors = [protected_vector_1, protected_vector_2, protected_vector_3]

    aggregated_vector = lom_1.aggregate(pvectors)
    sum_x = np.sum(np.array([params, params, params]), axis=0)
    assert aggregated_vector == sum_x.tolist()

    with pytest.raises(FedbiomedError) as e_info:
        r_int = random.getrandbits(32)
        params = [r_int, r_int]
        protected_vector_1 = lom_1.protect(node_ids[0], pwkeys[0], tau, params, node_ids)



@pytest.fixture
def lom_crypter(pairwise_keys):


    _, nonce, _ = pairwise_keys
    return SecaggLomCrypter(nonce)



def test_secaggg_lom_crypter_01_encrypt(lom_crypter, pairwise_keys):

    node_ids, nonce, pwkeys = pairwise_keys
    params = [1.5, 1.5, 1.5, 1.5, 1.5]
    round_ = 1

    encrypt = functools.partial(lom_crypter.encrypt,
        current_round = round_,
        node_ids=node_ids,
        params = params,
        weight=1
    )

    e1 = encrypt(node_id=node_ids[0], pairwise_secrets=pwkeys[0])
    e2 = encrypt(node_id=node_ids[1], pairwise_secrets=pwkeys[1])
    e3 = encrypt(node_id=node_ids[2], pairwise_secrets=pwkeys[2])


    result = lom_crypter.aggregate(
           [e1, e2, e3], 3
    )

    assert all(math.isclose(v1, v2, rel_tol = 0.0001)  for v1, v2 in zip(result, params))







if __name__ == "__main__":
    pytest.main()
