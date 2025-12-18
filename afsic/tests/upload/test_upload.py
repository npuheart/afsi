
from afsic import swanlab_init, swanlab_upload
import numpy as np
config = {
    'project_name': 'AFSI_upload_tests',
    'experiment_name': 'upload_tests_experiment_1',
}

swanlab_init(
    config['project_name'], 
    config['experiment_name'], 
    config,
    api_key='n9VCvqOuopPtCOVg6xJnB',
    host='http://swanlab.pengfeima.cn'
    )

data_log_1 = {}
for t in np.linspace(0, 0.1, 5):
    data_log_1["t"] = 10*t
    swanlab_upload(t, data_log_1, some_key = t)
