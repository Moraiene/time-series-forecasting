import os
from models.mlp.mlp import mlp
from models.lstm.lstm import lstm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def models():
    mlp()
    lstm()


if __name__ == '__main__':
    models()
