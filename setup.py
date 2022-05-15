import os
import shutil

from setuptools import setup
from Cython.Build import cythonize


file_name = "board_speedup"
setup(ext_modules = cythonize(file_name + ".pyx"))

# C 言語に変換して、コンパイルする過程でできた作業ファイル、ディレクトリを削除する
os.remove(file_name + ".cpp")
shutil.rmtree("build/")

print("\nSuccessfully creating so-file!")