from os.path import dirname, join, basename, exists
from os import remove
from glob import glob
from tempfile import mkdtemp
from shutil import move, rmtree

from setuptools import Extension, setup
from numpy import get_include
from Cython.Build import cythonize



# 第１引数はファイル名、第２引数は HTML ファイルを生成するかどうかの真偽値
def main(file_name = "speedup", annotate_flag = False):
    src_dir = dirname(__file__)
    src_file = join(src_dir, file_name)
    build_path = join(src_dir, "..", "..", "{}")
    new_exec_file = build_path.format(f"{file_name}.*")


    # ルートディレクトリに存在する、実行ファイルを抽出する条件に合致する全てのファイルを一旦退避させる
    duplicated_files = glob(new_exec_file)
    if duplicated_files:
        tmp_dir = mkdtemp()
        tmp_paths = []

        for duplicated_file in duplicated_files:
            tmp_path = join(tmp_dir, basename(duplicated_file))
            tmp_paths.append(tmp_path)
            move(duplicated_file, tmp_path)


    # セットアップ
    ext = Extension(file_name, sources = [f"{src_file}.pyx"],
                    include_dirs = [".", get_include()],
                    define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
    setup(name = file_name, ext_modules = cythonize([ext], annotate = annotate_flag))


    # 実行ファイルを src/pyx に移す
    new_exec_file = glob(new_exec_file).pop()
    old_exec_file = join(src_dir, basename(new_exec_file))

    if exists(old_exec_file):
        remove(old_exec_file)
    move(new_exec_file, src_dir)


    # 不要なファイル・ディレクトリを削除する
    remove(f"{src_file}.cpp")
    rmtree(build_path.format("build"))


    # 退避していたファイルを元に戻し、一時ディレクトリを削除する
    if duplicated_files:
        for tmp_path, duplicated_file in zip(tmp_paths, duplicated_files):
            move(tmp_path, duplicated_file)
        rmtree(tmp_dir)


    # 画面表示
    print("\nSuccessfully creating so-file!")


if __name__ == "__main__":
    main()