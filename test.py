import scipy.io as io
data = io.loadmat('practice.mat')
ranges = data['ranges']
angles = data['scanAngles']
pose = data['pose']

