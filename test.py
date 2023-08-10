import h5py
import torch
import sys
from BiMPan import BiMPan
import numpy as np
import scipy.io as sio
device=torch.device('cuda:1')
satellite = 'wv3'
model = BiMPan().to(device)
model.load_state_dict(torch.load('weights/BiMPan_500.pth'))

file_path = './data/test_reduce.h5'

def load_test_data(file_path):
    dataset = h5py.File(file_path, 'r')
    ms = np.array(dataset['ms'], dtype=np.float32) / 2047.0
    lms = np.array(dataset['lms'], dtype=np.float32) / 2047.0
    pan = np.array(dataset['pan'], dtype=np.float32) / 2047.0
    gt = np.array(dataset['gt'], dtype=np.float32) / 2047

dataset = h5py.File(file_path, 'r')
print(dataset)
print(type(dataset))

ms = np.array(dataset['ms'], dtype=np.float32) / 2047.0
lms = np.array(dataset['lms'], dtype=np.float32) / 2047.0
pan = np.array(dataset['pan'], dtype=np.float32) / 2047.0
gt = np.array(dataset['gt'], dtype=np.float32)

ms = torch.from_numpy(ms).float().to(device)
lms = torch.from_numpy(lms).float().to(device)
pan = torch.from_numpy(pan).float().to(device)

model.eval()

with torch.no_grad():
    import time
    t1 = time.time()
    out = model(pan, lms)
    t2 = time.time()
    print('time:{}s'.format((t2 - t1)))
    I_SR = torch.squeeze(out * 2047).cpu().detach().numpy()  # HxWxC
    I_MS_LR = torch.squeeze(ms * 2047).cpu().detach().numpy()  # HxWxC
    I_MS = torch.squeeze(lms * 2047).cpu().detach().numpy()  # HxWxC
    I_PAN = torch.squeeze(pan * 2047).cpu().detach().numpy()  # HxWxC
    I_GT = gt # HxWxC
    sio.savemat('./result/' + satellite + '.mat',
                {'I_SR': I_SR, 'I_MS_LR': I_MS_LR, 'I_MS': I_MS, 'I_PAN': I_PAN, 'I_GT': I_GT})