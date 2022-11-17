from setuptools import Extension, setup
import os
import argparse
from Cython.Distutils import build_ext
from Cython.Compiler.Options import get_directive_defaults
from Cython.Build import cythonize

from rich import print

parser = argparse.ArgumentParser()

parser.add_argument('--profile', action='store_true')
parser.add_argument('--hardcore', action='store_true')

args = parser.parse_args()

# Set the working directory
old_dir = os.getcwd()
packageDir = os.path.dirname(__file__)
includedDir = [packageDir]
os.chdir(packageDir)

extension_kwargs = dict( 
        include_dirs=includedDir,
        libraries=["m"],                # Unix-like specific link to C math libraries
        extra_compile_args=["-fopenmp"],# Links OpenMP for parallel computing
        extra_link_args=["-fopenmp"],
        )

cython_compiler_directives = get_directive_defaults()
cython_compiler_directives['language_level'] = "3"
cython_compiler_directives['warn'] = True

# Profiling using line_profiler
if args.profile:
    print("[blue]Compiling in [green]PROFILE[/green] mode[/blue]")
    cython_compiler_directives['profile'] = True
    cython_compiler_directives['linetrace'] = True
    cython_compiler_directives['binding'] = True
    # Activates profiling
    extension_kwargs["define_macros"] = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]

# Globally boost speed by disabling checks
# see https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
if args.hardcore:
    print("[blue]Compiling in [green]HARDCORE[/green] mode[/blue]")
    cython_compiler_directives['boundscheck'] = False
    cython_compiler_directives['cdivision'] = True
    cython_compiler_directives['wraparound'] = False

# Finds each .pyx file and adds it as an extension
cython_files = [file for file in os.listdir(".") if file.endswith(".pyx")]
print(f"Found {len(cython_files)} cython files ({cython_files})")
ext_modules = [
    Extension(
        cfile.strip(".pyx"),
        [cfile],
        **extension_kwargs
    )
    for cfile in cython_files
]

# Adds the random number generator
ext_modules.append(Extension("gigarand", ["gigarand/gigarand.c"]))

print(f"[blue]COMPILER DIRECTIVES[/blue]: {cython_compiler_directives}")
print(f"[blue]EXT_KWARGS[/blue]: {extension_kwargs}")
setup(
    name=packageDir,
    cmdclass={"build_ext": build_ext},
    include_dirs=includedDir,
    ext_modules=cythonize(  ext_modules, 
                            compiler_directives=cython_compiler_directives,
                            annotate=False),
                            script_args=["build_ext"],
                            options={"build_ext": {"inplace": True, "force": True}},
                        )

# Sets back working directory to old one
os.chdir(old_dir)
