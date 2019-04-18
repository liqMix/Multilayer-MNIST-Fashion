import numpy as np
import struct as st


# Copies the MNIST data into memory
class Dataset():
    def __init__(self):
        self.data = []
        self.labels = None


        with open('train-images-idx3-ubyte', mode='rb') as file:
            file.seek(0)

            magic = st.unpack('>4B', file.read(4))
            nImg = st.unpack('>I', file.read(4))[0]
            nR = st.unpack('>I', file.read(4))[0]
            nC = st.unpack('>I', file.read(4))[0]

            nBytesTotal = nImg * nR * nC * 1  # since each pixel data is 1 byte

            data = np.asarray(st.unpack('>' + 'B' * nBytesTotal, file.read(nBytesTotal))).reshape((nImg, nR, nC))
            
            for d in data:
                self.data.append(np.append((np.ravel(d) / 255), 1))

            self.data = np.array(self.data)

        with open('train-labels-idx1-ubyte', mode='rb') as file:
            file.seek(0)

            magic = st.unpack('>4B', file.read(4))
            nR = st.unpack('>I', file.read(4))[0]

            self.labels = np.zeros((nR, 1))
            nBytesTotal = nR * 1
            self.labels = np.asarray(st.unpack('>' + 'B' * nBytesTotal, file.read(nBytesTotal))).reshape((nR, 1))
