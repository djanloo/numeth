"""A dummy utils module.

Used to check if pyx-pyx imports are done correctly.
Remember that cdef makes stuff invisible outside module."""

cpdef urushibara_ruka(n):
    for _ in range(n):
        print("Daga.. otoko da")