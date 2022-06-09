import os
import shutil

from setuptools import Extension, setup
from numpy import get_include
from Cython.Build import cythonize



# HTML-file を作成するか否か
annotate_flag = False


# ファイルパス
file_name = "speedup"
dir_path = os.path.dirname(__file__) + os.sep
file_path = dir_path + file_name


# セットアップ
ext = Extension(file_name, sources = [f"{file_path}.pyx"],
                include_dirs = [".", get_include()],
                define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
setup(name = file_name, ext_modules = cythonize([ext], annotate = annotate_flag))


# 終了処理
os.remove(f"{file_path}.cpp")
shutil.rmtree(dir_path + "build" + os.sep)


# 画面表示
print("\nSuccessfully creating so-file!")