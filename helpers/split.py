from numpy import array


def split(sequence, steps):
    row_sequence, row_sequence_result = list(), list()
    for i in range(len(sequence)):
        row_sequence_len = i + steps
        if row_sequence_len > len(sequence) - 1:
            break
        row_sequence_list, row_sequence_result_list = sequence[i:row_sequence_len], sequence[row_sequence_len]
        row_sequence.append(row_sequence_list)
        row_sequence_result.append(row_sequence_result_list)
    return array(row_sequence), array(row_sequence_result)
