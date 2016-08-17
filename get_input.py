import numpy as np

arr = np.loadtxt('result.txt')
tot = 1500
inps = np.asarray([])

for c in range(0, 6):
    try:
        inps = np.vstack((inps, (arr[arr[:,0] == c])[:tot]))
    except:
        inps = (arr[arr[:,0] == c])[:tot]

np.savetxt('input.txt', inps, fmt='%.6g')
