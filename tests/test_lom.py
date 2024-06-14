import pytest
import numpy as np
from fedbiomed.common.secagg._lom import PRF, LOM  # Assuming the class definitions are in lom.py

def test_prf():
    nonce = b'\x00' * 16
    prf = PRF(nonce)

    pairwise_secret = b'\x01' * 32
    round = 1
    key = prf.eval_key(pairwise_secret, round)
    assert len(key) == 32

    seed = key
    input_size = 10
    buffer = prf.eval_vector(seed, round, input_size)
    vector = np.frombuffer(buffer, dtype='uint32')
    assert len(vector) == input_size
    

def test_lom():
    nonce = b'\x00' * 16
    lom_1 = LOM(nonce)
    lom_2 = LOM(nonce)
    lom_3 = LOM(nonce)
    nodes_ids = ['node1', 'node2', 'node3']
    lom_1.setup_pairwise_secrets(my_node_id='node1', nodes_ids=nodes_ids)
    lom_2.setup_pairwise_secrets(my_node_id='node2', nodes_ids=nodes_ids)
    lom_3.setup_pairwise_secrets(my_node_id='node3', nodes_ids=nodes_ids)



    
    tau = 1
    x_1_tau = [1, 2, 3, 4, 5]
    node_ids = ['node1', 'node2', 'node3']
    protected_vector_1 = lom_1.protect(tau, x_1_tau, node_ids)
    assert len(protected_vector_1) == len(x_1_tau)

    x_2_tau = [1, 2, 3, 4, 5]
    protected_vector_2 = lom_2.protect(tau, x_2_tau, node_ids)
    assert len(protected_vector_2) == len(x_2_tau)

    x_3_tau = [1, 2, 3, 4, 5]
    protected_vector_3 = lom_3.protect(tau, x_3_tau, node_ids)
    assert len(protected_vector_3) == len(x_3_tau)

    list_y_u_tau = [protected_vector_1, protected_vector_2, protected_vector_3]
    aggregated_vector = lom_1.aggregate(list_y_u_tau)
    assert len(aggregated_vector) == len(x_1_tau)
    assert aggregated_vector == [3, 6, 9, 12, 15]

if __name__ == "__main__":
    pytest.main()
