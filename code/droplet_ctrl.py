import jax
import jax.numpy as np
from jax import jit, grad, vmap, jacrev, jacfwd
from jax import random
from functools import partial
import time

displacement, shift = space.free()
metric = space.metric(displacement)
displacement = space.map_product(displacement)
metric = space.map_product(metric)

alpha = 100
# eta_1, eta_2 = 1e-2, 1e-8
N, dim = 100, 2
D_R = 0.5
t0, t1, dt = 0, 0.1, 0.001
steps = int((t1-t0)/dt)

box_size = box_size_at_number_density(particle_count = N, number_density = N/10, spatial_dimension = dim)
R0 = random.uniform(random.PRNGKey(0), (N, dim), maxval=box_size)

n = 100
xmin = np.min(R0[:,0]) - 0.5
xmax = np.max(R0[:,0]) + 0.5
ymin = np.min(R0[:,1]) - 0.5
ymax = np.max(R0[:,1]) + 0.5
X, Y = np.mgrid[xmin:xmax:(n*1j), ymin:ymax:(n*1j)]
positions = np.transpose(np.vstack([X.ravel(), Y.ravel()]))
positions_list = np.reshape(positions, (n*n,1,2))

def phi_field(x):
    return np.einsum('i->', np.exp(np.squeeze(- alpha*np.square(metric(x, R0)))))

batch_phi_field = vmap(phi_field)

fig, ax = plt.subplots()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.grid(False)
ax.set_aspect('equal', adjustable='box')
ax.pcolormesh(X, Y, np.reshape(batch_phi_field(positions_list), (n,n)), cmap='gist_heat')

plt.show()

def square_lattice(N, box_size):
  Nx = int(np.sqrt(N))
  Ny, ragged = divmod(N, Nx)
  if Ny != Nx or ragged:
    assert ValueError('Particle count should be a square. Found {}.'.format(N))
  length_scale = box_size / (Nx-1)
  R = []
  for i in range(Nx):
    for j in range(Ny):
      R.append([i * length_scale, j * length_scale])
  return np.array(R)

@jit
def lub_energy(R):
    surf_tension = 2e0
    eta = 1e1
    beta = 10
    dx2 = ((xmax-xmin)/n)**2

    def phi_field(x):
        return np.einsum('i->', np.exp(np.squeeze(- alpha*np.square(metric(x, R)))))

    def grad_phi_field(x):
        return jacrev(phi_field, argnums=0)(x)

    batch_phi_field = vmap(phi_field)
    batch_grad_phi_field = vmap(grad_phi_field)

    return 5e-2*dx2*np.einsum('i->', -((batch_phi_field(positions_list))**2 - eta)*(np.exp(-beta*(batch_phi_field(positions_list))**2)) + surf_tension*np.einsum('ij->i', np.square(np.squeeze(batch_grad_phi_field(positions_list)))))

@jit
def terminal_cost_L2(R, phi_target):
    def phi_field(x):
        return np.einsum('i->', np.exp(np.squeeze(- alpha*np.square(metric(x, R)))))
    batch_phi_field = vmap(phi_field)
    phi_eval_euler = batch_phi_field(positions_list)
    return np.sum(np.square(phi_eval_euler - phi_target))

import jax
import jax.numpy as jnp
from jax import jit, vmap

@jit
def sinkhorn_distance(a, b, M, eps=0.01, max_iter=100):
    """
    Compute Sinkhorn distance between two distributions
    a, b: 1D arrays representing the two distributions
    M: cost matrix
    eps: regularization parameter
    max_iter: maximum number of iterations
    """
    K = jnp.exp(-M / eps)

    def step(carry, _):
        u, v = carry
        u = a / (K @ v)
        v = b / (K.T @ u)
        return (u, v), None

    u = jnp.ones_like(a)
    v = jnp.ones_like(b)
    (u, v), _ = jax.lax.scan(step, (u, v), None, length=max_iter)

    return jnp.sum(u * (K @ v) * M)

@jit
def terminal_cost_W2(R, phi_target):
    phi_target_reshaped = phi_target.reshape(20, 5, 20, 5)
    phi_target = np.mean(phi_target_reshaped, axis=(1, 3))

    n = 20  # Increase this for higher resolution
    x = jnp.linspace(0, box_size, n)
    y = jnp.linspace(0, box_size, n)
    X, Y = jnp.meshgrid(x, y)
    positions = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # Compute the density field for the current droplet configuration
    def phi_field(pos):
        diff = pos[jnp.newaxis, :] - R[:, jnp.newaxis, :]
        distances_squared = jnp.sum(diff**2, axis=-1)
        return jnp.sum(jnp.exp(-alpha * distances_squared))

    batch_phi_field = vmap(phi_field)
    phi_eval = batch_phi_field(positions)

    # Reshape and normalize distributions
    phi_eval = phi_eval.reshape(-1)
    phi_target = phi_target.reshape(-1)

    phi_eval_norm = phi_eval / (jnp.sum(phi_eval) + 1e-10)
    phi_target_norm = phi_target / (jnp.sum(phi_target) + 1e-10)

    # Compute cost matrix
    X_flat = X.ravel()[:, None]
    Y_flat = Y.ravel()[:, None]
    M = jnp.sqrt((X_flat - X_flat.T)**2 + (Y_flat - Y_flat.T)**2)

    # # You can also include an L2 term if desired
    # l2_distance = jnp.sum(jnp.square(phi_eval - phi_target))

    return sinkhorn_distance(phi_eval_norm, phi_target_norm, M)

@partial(jit, static_argnums=(2,))
def controlled_simulate(R0, phi_target, steps):

    def single_step_control(R):
        # Forward pass
        def forward_step(carry, _):
            R, key = carry
            key, subkey = random.split(key)
            R_force = -grad(lub_energy)(R)
            R_new = R + dt * R_force + np.sqrt(2*D_R*dt) * random.normal(subkey, R.shape)
            return (R_new, key), R_new

        init_key = random.PRNGKey(0)
        _, R_list = jax.lax.scan(forward_step, (R, init_key), np.arange(int(steps/30)))

        # Compute final lambda
        lambda_R_final = 1.0*grad(terminal_cost_L2, 0)(R_list[-1], phi_target) + 0.1*grad(terminal_cost_W2, 0)(R_list[-1], phi_target)

        # Backward pass
        def backward(R, lambda_R):
            R_force = -grad(lub_energy)(R)
            R_new = R - dt * R_force

            # Compute the gradients of the forward dynamics
            def forward_dynamics(R):
                return -grad(lub_energy)(R)

            R_grad = jacfwd(forward_dynamics)(R)

            # Update lambda_R using the adjoint equation
            lambda_R_new = lambda_R - dt * np.einsum('ijkl,kl->ij', R_grad, lambda_R)

            return R_new, lambda_R_new

        def backward_step(carry, _):
            R, lambda_R = carry
            R_new, lambda_R_new = backward(R, lambda_R)
            return (R_new, lambda_R_new), (R_new, lambda_R_new)

        _, (R_list, lambda_R_list) = jax.lax.scan(backward_step, (R_list[-1], lambda_R_final), np.arange(int(steps/30)))

        # Use the initial lambda (which is now at the end of our reversed list)
        lambda_R_initial = lambda_R_list[-1]
        u_R = -0.5 * lambda_R_initial

        return u_R

    def body_fun(carry, _):
        R, key = carry
        key, subkey = random.split(key)

        # Compute control for this step
        u_R = single_step_control(R)

        # Apply lubrication force
        R_force = -grad(lub_energy)(R)

        # Update R with lubrication, control, and noise
        R_new = R + dt * (R_force + u_R) + np.sqrt(2*D_R*dt) * random.normal(subkey, R.shape)

        return (R_new, key), (R_new, u_R)

    init_key = random.PRNGKey(0)
    _, (R_list, u_R_list) = jax.lax.scan(body_fun, (R0, init_key), np.arange(steps))
    return R_list, u_R_list

# Load initial and target states
# R_init = np.load('R_init.npy', allow_pickle=True)
phi_target = np.load('phi_target.npy', allow_pickle=True)

# Run controlled simulation
controlled_R_list, controlled_u_R_list = controlled_simulate(controlled_R_list[-1], phi_target, steps)

controlled_R_list_save = []
controlled_R_list_save.append(controlled_R_list)

# Animate results
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

fig_size = (10,10)
fig, ax = plt.subplots(figsize=fig_size)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.grid(False)
ax.set_aspect('equal', adjustable='box')

def animate(j):
    ax.clear()
    R = controlled_R_list[j]
    def phi_field(x):
        return np.einsum('i->', np.exp(np.squeeze(- alpha*np.square(metric(x, R)))))
    batch_phi_field = vmap(phi_field)
    phi_eval_euler = np.reshape(batch_phi_field(positions_list), (n,n))
    fgrnd1 = ax.pcolormesh(X, Y, phi_eval_euler)
    return fgrnd1

ani = FuncAnimation(fig, animate, interval=50, blit=False, repeat=True, frames=len(controlled_R_list))
ani.save("droplet_controlled_adjoint_3.gif", dpi=300, writer=PillowWriter(fps=10))
