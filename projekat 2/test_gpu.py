import sys
sys.path.insert(0, r"D:\Python_packages")
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))