import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import animation

# ----- Toy 2D data (so we can visualize beta1, beta2) -----
np.random.seed(7)
M = 100
beta_true = np.array([1.2, 0.0])
X = np.random.randn(M, 2)
y = X @ beta_true + 0.25*np.random.randn(M)

# Center features and target
X -= X.mean(axis=0, keepdims=True)
y -= y.mean()

lam = 0.6

def lasso_obj(beta):
    resid = y - X @ beta
    return (resid @ resid) / M + lam * (np.abs(beta).sum())

def subgrad(beta):
    grad_smooth = (2.0 / M) * (X.T @ (X @ beta - y))
    sgn = np.sign(beta)
    sgn[beta == 0.0] = 0.0  # valid choice in [-1,1]
    return grad_smooth + lam * sgn

# ----- Subgradient descent -----
beta = np.array([0.0, 0.0])
steps = [beta.copy()]
T = 80
eta0 = 0.8
for t in range(1, T+1):
    g = subgrad(beta)
    eta = eta0 / np.sqrt(t)
    beta = beta - eta * g
    steps.append(beta.copy())
steps = np.array(steps)
Z_path = np.array([lasso_obj(b) for b in steps])

# ----- Build grid for surface -----
pad = 1.0
b1_min, b1_max = steps[:,0].min() - pad, steps[:,0].max() + pad
b2_min, b2_max = steps[:,1].min() - pad, steps[:,1].max() + pad
b1 = np.linspace(b1_min, b1_max, 160)
b2 = np.linspace(b2_min, b2_max, 160)
B1, B2 = np.meshgrid(b1, b2)
Z = np.zeros_like(B1)
for i in range(B1.shape[0]):
    for j in range(B1.shape[1]):
        Z[i, j] = lasso_obj(np.array([B1[i, j], B2[i, j]]))

# ----- Static 3D surface + path (improved visibility) -----
eps = 0.02 * (Z.max() - Z.min())  # small z-offset to lift the path above surface

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(B1, B2, Z, linewidth=0, antialiased=True, alpha=0.35)  # more transparent
ax.plot3D(steps[:,0], steps[:,1], Z_path + eps, marker="o", linewidth=2, markersize=5)
# helpful projection on the bottom plane
ax.contour(B1, B2, Z, 20, zdir='z', offset=Z.min(), linewidths=0.8, alpha=0.5)

ax.set_title("LASSO Objective Surface + Subgradient Descent Path (3D)")
ax.set_xlabel(r"$\beta_1$")
ax.set_ylabel(r"$\beta_2$")
ax.set_zlabel("Objective")
ax.view_init(elev=32, azim=-40)  # tweak the view so points pop out
plt.tight_layout()
plt.savefig("lasso_3d_surface_path.png", dpi=180)
plt.show()

# ----- Animation: path growing (improved visibility) -----
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection="3d")
ax2.plot_surface(B1, B2, Z, linewidth=0, antialiased=True, alpha=0.35)
line, = ax2.plot([], [], [], marker="o", linewidth=2, markersize=5)
pt = ax2.scatter([], [], [], s=40)  # current point as a bigger dot
ax2.set_title("Subgradient Descent on LASSO Surface (3D Animation)")
ax2.set_xlabel(r"$\beta_1$")
ax2.set_ylabel(r"$\beta_2$")
ax2.set_zlabel("Objective")
ax2.view_init(elev=32, azim=-40)

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    # init current point
    pt._offsets3d = ([], [], [])
    return (line, pt)

def update(frame):
    line.set_data(steps[:frame,0], steps[:frame,1])
    line.set_3d_properties(Z_path[:frame] + eps)

    # update current point
    if frame > 0:
        b1v = steps[frame-1, 0]
        b2v = steps[frame-1, 1]
        zv  = Z_path[frame-1] + eps
        pt._offsets3d = ([b1v], [b2v], [zv])
    return (line, pt)

ani = animation.FuncAnimation(fig2, update, init_func=init,
                              frames=len(steps), interval=120, blit=True)

ani.save("lasso_3d_path.gif", writer="pillow", fps=8)
plt.close(fig2)
