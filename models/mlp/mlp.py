from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from helpers.split import split
from helpers.settings import steps
from helpers.settings import samples
from data.rows import row
from data.rows import predict


def model_mlp(row_sequence, row_sequence_result, row_sequence_in):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=steps))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(row_sequence, row_sequence_result, epochs=2000, verbose=0)
    return model.predict(row_sequence_in, verbose=0)


def mlp():
    row_sequence, row_sequence_result = split(row, steps)
    row_sequence_in = array(predict)
    row_sequence_in = row_sequence_in.reshape((samples, steps))
    result = model_mlp(row_sequence, row_sequence_result, row_sequence_in)
    print('MLP:', result)
