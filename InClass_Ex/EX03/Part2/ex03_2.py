def is_prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

# simple fix to handle edge cases
def fix_is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True