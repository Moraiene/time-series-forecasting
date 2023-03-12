import os
from models.mlp.mlp import mlp
from models.lstm.lstm import lstm
from models.cnn.cnn import cnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def models():
    mlp()
    lstm()
    cnn()


if __name__ == '__main__':
    models()
