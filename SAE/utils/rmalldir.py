import shutil


def rmalldir(path):
    shutil.rmtree(path, ignore_errors=False, onerror=None)
