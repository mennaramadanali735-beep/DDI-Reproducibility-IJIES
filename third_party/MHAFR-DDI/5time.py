import os, sys
import torch
import random
import numpy as np

def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=False)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
python_exe = sys.executable
for k in range(5):

    os.system(f'{python_exe} ddi_dnn-z.py --out_file test.txt --zhongzi '+str(k))



