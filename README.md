# cythonTemplate
Personal template for a generic cython project setup

- Activate environment (``pipenv shell``)
- Run ``make`` or possibly (hehe) ``import setup`` in the code to rebuild the package

## Benchmark

The benchmark problem is counting the number of primes up to a certain value.
Trivial algorithm chosen:
  - First three stop at n
  - Last three (_r_) stop at ``sqrt(n)``
 
 This allows to compare the ``sqrt`` evaluation (C vs Python).
 
 Parallel computing using ``openMP`` is tested too, releasing the GIL and multithreading.
 
## Results
### O(n^2) functions
Just for declaring types and changing from ``range`` to ``prange`` we get:
  - Serial: **x12**
  - Parallel: **x24**
### O(n^3/2) functions
Using ``libc.math``:
  - Serial: **x900**
  - Parallel: **x1700**
 
![Screenshot from 2022-09-01 16-00-16](https://user-images.githubusercontent.com/89815653/187933094-e6dd8714-0b74-4fed-9cbb-391c61ce5aa0.png)

Overall boost: **3000x**

## Profiling
Profiling slows down the code, so the compiler is not automatically set in profiler mode.

When needed, rebuild the package using

``make profile``

An example of line profiling is given in ``profile.py``.

Some bugs in ``line_profiler`` for cython, not all functions are profiled.

A workaround is to shift the missing functions at the top of the code.

## Other compiling modes
  - ``make hardcore``: globally disables array wrapping, bound checks and enables cdivision
  - ``make notrace``: manually disables profiling

![Screenshot from 2022-09-04 13-46-44](https://user-images.githubusercontent.com/89815653/188322063-4bd82d34-6767-4a0c-8ae1-5c219e839862.png)

![Screenshot from 2022-09-04 13-47-47](https://user-images.githubusercontent.com/89815653/188322066-f5ec23a7-cd15-4bab-8926-e90dbea7aa8b.png)

