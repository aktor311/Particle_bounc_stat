import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib.animation import PillowWriter
import scienceplots
plt.style.use(['science', 'notebook'])
from itertools import combinations

def get_delta_pairs(x, ids_pairs):
    return np.diff(np.array([x[ids_pairs[:, 0]], x[ids_pairs[:, 1]]]).T, axis=1).ravel()


def get_deltad_pairs(r, ids_pairs):
    return np.sqrt(get_delta_pairs(r[0], ids_pairs) ** 2 + get_delta_pairs(r[1], ids_pairs) ** 2)


def compute_new_v(v1, v2, r1, r2):
    v1new = v1 - ((v1 - v2) * (r1 - r2)).sum(axis=0) / np.sum((r1 - r2) ** 2, axis=0) * (r1 - r2)
    v2new = v2 - ((v1 - v2) * (r1 - r2)).sum(axis=0) / np.sum((r2 - r1) ** 2, axis=0) * (r2 - r1)
    return v1new, v2new


def motion(r, v, id_pairs, ts, dt, d_cutoff):
    rs = np.zeros((ts, r.shape[0], r.shape[1]))
    vs = np.zeros((ts, v.shape[0], v.shape[1]))
    # Initial State
    rs[0] = r.copy()
    vs[0] = v.copy()
    for i in range(1, ts):
        ic = id_pairs[get_deltad_pairs(r, ids_pairs) < d_cutoff]
        v[:, ic[:, 0]], v[:, ic[:, 1]] = compute_new_v(v[:, ic[:, 0]], v[:, ic[:, 1]], r[:, ic[:, 0]], r[:, ic[:, 1]])

        v[0, r[0] > 1] = -np.abs(v[0, r[0] > 1])
        v[0, r[0] < 0] = np.abs(v[0, r[0] < 0])
        v[1, r[1] > 1] = -np.abs(v[1, r[1] > 1])
        v[1, r[1] < 0] = np.abs(v[1, r[1] < 0])

        r = r + v * dt
        rs[i] = r.copy()
        vs[i] = v.copy()
    return rs, vs

n_particles = 5000
r = np.random.random((2,n_particles))
ixr = r[0]>0.5
ixl = r[0]<=0.5
ids = np.arange(n_particles)
ids_pairs = np.asarray(list(combinations(ids,2)))
v = np.zeros((2,n_particles))
v[0][ixr] = -500
v[0][ixl] = 500
radius = 0.0015
rs, vs = motion(r, v, ids_pairs, ts=1000, dt=0.000008, d_cutoff=2*radius)

v = np.linspace(0, 2000, 1000)
a = 2/500**2
fv = a*v*np.exp(-a*v**2 / 2)

bins = np.linspace(0,1500,50)
plt.figure()
plt.hist(np.sqrt(np.sum(vs[400]**2, axis=0)), bins=bins, density=True)
plt.plot(v,fv)
plt.xlabel('Velocity [m/s]')
plt.ylabel('# Particles')
plt.show()