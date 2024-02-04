from setuptools import setup, find_packages, Extension
import pathlib
import os
from os.path import (join,
                     relpath,
                     splitext)


def create_cython_ext(folder: pathlib.Path, **other_args):

    sources = [str(folder / file) for file in folder.iterdir()
               if file.suffix == '.pyx']

    rel_paths = [relpath(source, join(os.getcwd(), "src"))
                 for source in sources]

    names = [splitext(path)[0].replace('/', '.') for path in rel_paths]
    from numpy import get_include
    include_dirs = [get_include()]
    macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    if 'include_dirs' in other_args:
        other_args['include_dirs'] += include_dirs
    else:
        other_args['include_dirs'] = include_dirs

    if 'define_macros' in other_args:
        other_args['define_macros'] += macros
    else:
        other_args['define_macros'] = macros

    ext_list = []
    for name, source in zip(names, sources):
        ext_list.append(Extension(name=name,
                                  sources=[source],
                                  **other_args))

    return ext_list


def main():
    here = pathlib.Path(__file__).parent
    print(here, os.getcwd())
    name = "flowpy"
    version = "0.0.1"
    description = "Package to process structured 3D CFD simulation data"
    long_description = str(here / "README.md")
    long_description_content_type = "text/markdown"
    author = 'Matthew A. Falcone'
    packages = find_packages(where='src')
    package_dir = {"": 'src'}
    python_requires = ">=3.9, <4"
    install_requires = ['numpy',
                        'cython']
    extras_require = {"dev": []}
    test_suite = 'nose.collector'
    tests_require = ['nose']

    cython_dir = here / "src" / "flowpy" / "cython"
    cython_files = create_cython_ext(cython_dir,
                                     extra_compile_args=[
                                         "-fopenmp", "-O3", "-fopt-info-vec-missed"],
                                     extra_link_args=["-fopenmp", "-O3"])

    from Cython.Build import cythonize
    ext_list = cythonize(cython_files,
                         compiler_directives={'language_level': 3})

    print(name)

    setup(name=name,
          version=version,
          description=description,
          long_description=long_description,
          long_description_content_type=long_description_content_type,
          author=author,
          packages=packages,
          package_dir=package_dir,
          python_requires=python_requires,
          install_requires=install_requires,
          extras_require=extras_require,
          ext_modules=ext_list)


if __name__ == '__main__':
    main()
