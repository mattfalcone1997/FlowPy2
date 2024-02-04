from setuptools import setup, Extension
from Cython.Build import cythonize
from numpy import get_include
from os.path import (join,
                     relpath,
                     splitext)
from os import (listdir,
                getcwd)


def create_cython_ext(folder: str, **other_args) -> list[Extension]:

    sources = [join(folder, file) for file in listdir(folder)
               if splitext(file)[1] == '.pyx']

    rel_paths = [relpath(source, join(getcwd(), "src"))
                 for source in sources]

    names = [splitext(path)[0].replace('/', '.') for path in rel_paths]

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

    cython_dir = join("src", "flowpy", "cython")
    cython_files = create_cython_ext(cython_dir,
                                     extra_compile_args=[
                                         "-fopenmp", "-O3", "-fopt-info-vec-missed"],
                                     extra_link_args=["-fopenmp", "-O3"])

    ext_list = cythonize(cython_files,
                         compiler_directives={'language_level': 3})

    setup(ext_modules=ext_list)


if __name__ == '__main__':
    main()
