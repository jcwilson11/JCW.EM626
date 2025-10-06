from ex03_2 import is_prime
from ex03_2 import fix_is_prime

# step 1: 3 test cases
def test_is_prime():
    assert is_prime(2) == True
    assert is_prime(3) == True
    assert is_prime(4) == False

# step 2: at least one edge case the function fails on
'''
def test_is_prime_edge_case():
    # intentional fails
    assert is_prime(1) == False  # Edge case: 1 is not a prime number
    assert is_prime(0) == False  # Edge case: 0 is not a prime number
    assert is_prime(-5) == False # Edge case: negative numbers are not prim
'''

# step 3: simple fix to the function
def test_fix_is_prime():
    assert fix_is_prime(2) == True
    assert fix_is_prime(3) == True
    assert fix_is_prime(4) == False
    # edge cases
    assert fix_is_prime(1) == False  
    assert fix_is_prime(0) == False  
    assert fix_is_prime(-5) == False # Edge case: negative numbers are not prime