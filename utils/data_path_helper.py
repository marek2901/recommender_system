import os

def datasets_path():
    return os.path.join(
        os.path.dirname(__file__),
        '../datasets'
    )
