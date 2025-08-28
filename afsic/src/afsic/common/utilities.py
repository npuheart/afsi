from mpi4py import MPI
import swanlab, time
from datetime import datetime
import os

start_time = time.time()
dt_minimum = 1e-5


class TimeManager:
    def __init__(self, total_time, total_steps, fps=10):
        self.total_time = total_time  # 总时间T
        self.total_steps = total_steps  # 总步数
        self.fps = fps  # 每秒帧数
        self.time_per_step = total_time / total_steps  # 每步时间
        self.step_interval = max(1,int(total_steps / (fps * total_time)))  # 计算多少步输出一次

    def should_output(self, current_step):
        # 判断当前步是否是输出步
        if current_step > self.total_steps - 1:
            raise ValueError("当前步数超过总步数。")
        if current_step % int(self.step_interval) == 0:
            return True
        if current_step == self.total_steps - 1:
            return True
        return False


def swanlab_init(project_name, experiment_name, config):
    if (MPI.COMM_WORLD.rank == 0):
        swanlab.login(api_key="VBxEp1UBe2606KHDM9264", host='https://swanlab.cn', save=True)
        # swanlab.login(api_key='5z4lwzHZK8rpXY1lyTcey', host='http://swanlab.pengfeima.cn', save=True)
        swanlab.init(
            project=project_name,
            # workspace="deepheart",
            experiment_name=experiment_name,
            description="心肌分区域灌注模型",
            public=True,
            tags=["dolfinx", "dynamic", "hyperelastic", "swanlab"],
            config=config,
        )
    

def swanlab_upload(current_time, data_log_1, **params):
    data_log = {}
    data_log["time"] = current_time
    data_log["timecost"] = time.time() - start_time
    params = params or {}
    data_log.update(params)
    data_log.update(data_log_1)
    if (MPI.COMM_WORLD.rank == 0):
        swanlab.log(
            data_log, step = int(1+current_time/dt_minimum)
        )



output_path = "/home/dolfinx/afsi/data/"


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} 文件夹已创建。")
    else:
        print(f"{path} 文件夹已存在。")

def unique_filename(current_file_name, tag="normal"):
    check_path(output_path)
    note = os.path.splitext(current_file_name)[0]
    file_id = (
        f"{output_path}{note}/{tag}/"
        + datetime.now().strftime("%Y%m%d-%H%M%S") 
        + "/"
    )
    check_path(f"{output_path}{note}/{tag}/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    return file_id