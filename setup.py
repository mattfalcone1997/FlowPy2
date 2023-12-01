from setuptools import setup, find_packages
import pathlib
import os


def main():
    here = pathlib.Path(__file__).parent.resolve()

    name = "flowpy"
    version = "0.0.1"
    description = "Package to process structured 3D CFD simulation data"
    long_description = os.path.join(here, "README.md")
    long_description_content_type = "text/markdown"
    author = 'Matthew A. Falcone'
    packages = find_packages(where='.')
    python_requires = ">=3.7, <4"
    install_requires = ['numpy',
                        'matplotlib',
                        'numba',
                        'scipy']
    extras_require = {"dev": []}
    test_suite = 'nose.collector'
    tests_require = ['nose']

    setup(name=name,
          version=version,
          description=description,
          long_description=long_description,
          long_description_content_type=long_description_content_type,
          author=author,
          packages=packages,
          python_requires=python_requires,
          install_requires=install_requires,
          extras_require=extras_require)


if __name__ == '__main__':
    main()
