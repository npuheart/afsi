from mpi4py import MPI
import numpy as np
import hashlib
from dolfinx import log


def fun_fiber_v1(file_f0, file_s0, file_cdm):
    # 读取文本文件到 float64 类型的 NumPy 数组
    data_f0 = np.loadtxt(file_f0, dtype=np.float64).reshape(-1, 3)
    data_s0 = np.loadtxt(file_s0, dtype=np.float64).reshape(-1, 3)
    data_cdm = np.loadtxt(file_cdm, dtype=np.int64)
    dict_fibers = {}
    for data in zip(data_f0, data_s0, data_cdm):
        # print(data)
        f0, s0, cdm = data
        dict_fibers[cdm] = np.hstack((f0, s0))

    return dict_fibers

class CoordinateDataMap:
    def __init__(self):
        self.data_map = []
    
    def add_data(self, coord):
        coord_key = self.hash_floats(coord)
        self.data_map.append(coord_key)
        return coord_key

    def get_data(self, coord):
        coord_key = self.hash_floats(coord)
        if coord_key in self.data_map:
            return self.data_map[coord_key]
        else:
            log.error(f"Coordinate {coord} not found in data map.")
        return None

    def __contains__(self, coord):
        return self.hash_floats(coord) in self.data_map

    def hash_floats(self, coords, precision=6):
        # str_repr = f"{round(coords[0], precision):.{precision}f},{round(coords[1], precision):.{precision}f},{round(coords[2], precision):.{precision}f}"
        str_repr = f"{round(100*coords[0]+10*coords[1]+coords[2], precision):.{precision}f}"
        return int(hashlib.sha256(str_repr.encode()).hexdigest()[:8], 16)
