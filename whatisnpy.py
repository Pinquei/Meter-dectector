import numpy as np
from glob import glob
def load_data():
    data_type = "indicator_data"
    path = glob('./%s/%s/*' % (data_type,"test"))
    temp = np.load("./rg.npy").astype(int).tolist()
    print(temp)