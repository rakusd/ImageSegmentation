import numpy as np


class PmatrixParser:
    def parse(self, file_path: str) -> dict:
        with open(file_path, 'r') as file:
            dictionary = dict()
            for line in file.readlines():
                splitted = line.strip().split(' ')
                splitted[1:] = [float(s) for s in splitted[1:]]
                matrix = [
                    [splitted[1], splitted[2], splitted[3], splitted[4]],
                    [splitted[5], splitted[6], splitted[7], splitted[8]],
                    [splitted[9], splitted[10], splitted[11], splitted[12]]
                ]
                dictionary[splitted[0]] = np.array(matrix)
            return dictionary
