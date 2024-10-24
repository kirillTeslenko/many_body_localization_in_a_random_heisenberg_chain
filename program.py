import numpy as np
import scipy
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from datetime import datetime

Id = sparse.csr_matrix(np.eye(2))
Sx = sparse.csr_matrix([[0., 1.], [1., 0.]])
Sz = sparse.csr_matrix([[1., 0.], [0., -1.]])
Splus = sparse.csr_matrix([[0., 1.], [0., 0.]])
Sminus = sparse.csr_matrix([[0., 0.], [1., 0.]])


def singlesite_to_full(op, i, L):
    op_list = [Id]*L  # = [Id, Id, Id ...] with L entries
    op_list[i] = op
    full = op_list[0]
    for op_i in op_list[1:]:
        full = sparse.kron(full, op_i, format="csr")
    return full


def gen_sx_list(L):
    return [singlesite_to_full(Sx, i, L) for i in range(L)]


def gen_sz_list(L):
    return [singlesite_to_full(Sz, i, L) for i in range(L)]


def gen_hamiltonian(sx_list, sz_list, g, J=1.):
    L = len(sx_list)
    H = sparse.csr_matrix((2**L, 2**L))
    for j in range(L):
        H = H - J *( sx_list[j] * sx_list[(j+1)%L])
        H = H - g * sz_list[j]
    return H


def lanczos(psi0, H, N=200, stabilize=False):
    """Perform a Lanczos iteration building the tridiagonal matrix T and ONB of the Krylov space."""
    if psi0.ndim != 1:
        raise ValueError("psi0 should be a vector, "
                         "i.e., a numpy array with a single dimension of len 2**L")
    if H.shape[1] != psi0.shape[0]:
        raise ValueError("Shape of H doesn't match len of psi0.")
    psi0 = psi0/np.linalg.norm(psi0)
    vecs = [psi0]
    T = np.zeros((N, N))
    psi = H @ psi0  # @ means matrix multiplication
    # and works both for numpy arrays and scipy.sparse.csr_matrix
    alpha = T[0, 0] = np.inner(psi0.conj(), psi).real
    psi = psi - alpha* vecs[-1]
    for i in range(1, N):
        beta = np.linalg.norm(psi)
        if beta  < 1.e-13:
            print("Lanczos terminated early after i={i:d} steps:"
                  "full Krylov space built".format(i=i))
            T = T[:i, :i]
            break
        psi /= beta
        # note: mathematically, psi should be orthogonal to all other states in `vecs`
        if stabilize:
            for vec in vecs:
                psi -= vec * np.inner(vec.conj(), psi)
            psi /= np.linalg.norm(psi)
        vecs.append(psi)
        psi = H @ psi - beta * vecs[-2]
        alpha = np.inner(vecs[-1].conj(), psi).real
        psi = psi - alpha * vecs[-1]
        T[i, i] = alpha
        T[i-1, i] = T[i, i-1] = beta
    return T, vecs


def colorplot(xs, ys, data, **kwargs):
    """Create a colorplot with matplotlib.pyplot.imshow.

    Parameters
    ----------
    xs : 1D array, shape (n,)
        x-values of the points for which we have data; evenly spaced
    ys : 1D array, shape (m,)
        y-values of the points for which we have data; evenly spaced
    data : 2D array, shape (m, n)
        ``data[i, j]`` corresponds to the points ``(xs[i], ys[j])``
    **kwargs :
        additional keyword arguments, given to `imshow`.
    """
    data = np.asarray(data)
    if data.shape != (len(xs), len(ys)):
        raise ValueError("Shape of data doesn't match len of xs and ys!")
    dx = (xs[-1] - xs[0])/(len(xs)-1)
    assert abs(dx - (xs[1]-xs[0])) < 1.e-10
    dy = (ys[-1] - ys[0])/(len(ys)-1)
    assert abs(dy - (ys[1]-ys[0])) < 1.e-10
    extent = (xs[0] - 0.5 * dx, xs[-1] + 0.5 * dx,  # left, right
              ys[0] - 0.5 * dy, ys[-1] + 0.5 * dy)  # bottom, top
    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('extent', extent)
    # convention of imshow: matrix like data[row, col] with (0, 0) top left.
    # but we want data[col, row] with (0, 0) bottom left -> transpose and invert y axis
    plt.imshow(data.T[::-1, :], **kwargs)


# Additional functions to generate Hamiltonian for Néel state


def plot_E_vs_LanzcosIter(T):
    # Plot the results
    E = np.linalg.eigvalsh(T)
    Ns = np.arange(10, len(T))
    plt.figure(figsize=(13, 10))
    Es = []
    for Num in Ns:
        E = np.linalg.eigvalsh(T[:Num, :Num])
        Es.append(E[:10])

    plt.plot(Ns, Es)
    #plt.ylim(np.min(Es)-0.1, np.min(Es) + 5.)
    plt.title("stabilize")
    plt.xlabel("Lanczos iteration $N$")
    plt.ylabel("Energies")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"plot_EvsIterat_{current_time}.jpg"
    plt.savefig(filename)
    plt.show()

    return None
def generate_neel_state(L):

    length = 2 ** L
    vector = np.array([(i % 2) for i in range(length)])
    neel_state = np.roll(vector, 1)

    return neel_state / np.linalg.norm(neel_state)

def generate_random_field(L, W):
    return np.random.uniform(-W, W, size=L)

def gen_heisenberg_hamiltonian(sz_list, J=1., W=1.):
    L = len(sz_list)
    H = sparse.csr_matrix((2**L, 2**L))
    h = generate_random_field(L, W)
    for i in range(L):
        H += J * sz_list[i] @ sz_list[(i+1) % L]
        H -= h[i] * sz_list[i]
    return H

def time_evolve(psi0, H, t, N=200):
    """Evolve the state psi0 under Hamiltonian H for time t using Lanczos algorithm."""
    T, vecs = lanczos(psi0, H, N)
    eigvals, eigvecs = np.linalg.eigh(T)
    exp_T = np.diag(np.exp(-1j * eigvals * t))
    evolved_vecs = eigvecs @ exp_T @ eigvecs.T
    psi_t = np.zeros(psi0.shape, dtype=np.complex128)
    for i, vec in enumerate(vecs):
        psi_t += evolved_vecs[0, i] * vec
    return psi_t

def plot_time_evolve(T):
    eigvals, eigvecs = np.linalg.eigh(T)
    plt.plot(eigvals)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.show()
    return True

def reduced_density_matrix(psi, L):
    """Calculate the reduced density matrix for the first half of the chain."""
    psi = psi.reshape([2] * L)
    dim = 2**(L//2)
    psi = psi.transpose([i for i in range(0, L//2)] + [i for i in range(L//2, L)])
    psi = psi.reshape((dim, dim))
    rho = np.dot(psi, psi.conj().T)
    return rho

def entanglement_entropy(rho):
    """Calculate the entropy of a density matrix."""
    eigenvalues = np.linalg.eigvalsh(rho)
    entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-12))
    return entropy


def calculate_imbalance(psi, sz_list, L):
    sz_odd = sum([sz_list[i] for i in range(L) if i % 2 == 0])
    sz_even = sum([sz_list[i] for i in range(L) if i % 2 != 0])
    sz_odd_exp = np.vdot(psi, sz_odd @ psi).real
    sz_even_exp = np.vdot(psi, sz_even @ psi).real
    imbalance = sz_odd_exp - sz_even_exp
    return imbalance

if __name__ == "__main__":
    L = 14 # System size
    W = 5.0 # Disorder strength
    sx_list = gen_sx_list(L)
    sz_list = gen_sz_list(L)

    H = gen_heisenberg_hamiltonian(sz_list, J=1., W=W)
    psi0 = generate_neel_state(L)
    #print(psi0)


    # Uncomment out to get plot analogous to exercise 5
    # Perform Lanczos iteration
    #T, vecs = lanczos(psi0, H, N=200, stabilize=True)
    #plot_E_vs_LanzcosIter(T)


    # Generate time steps
    #time = []
    #t = 0.1
    #while t < 1:
    #    time.append(t)
    #    t += 0.1
    #t = 1
    #while t <= 50:
    #    time.append(t)
    #    t += 0.5
    #t = 10
    #while t <= 5000:
    #    time.append(t)
    #    t *= 1.5
    ###time = np.linspace(0,1000, 1)
    time = list(range(1, 1001))

    # Time evolution

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    psi_t = psi0
    plotE = []
    plotImba = []
    for t in time:
        psi_t_next = time_evolve(psi_t, H, t, N=200)

        rho = reduced_density_matrix(psi_t_next, L)
        entropy = entanglement_entropy(rho)
        plotE.append(entropy)
        print(f"Half-chain entanglement entropy at time t={t}: {entropy}")
        psi_t = psi_t_next
        imba = calculate_imbalance(psi_t, sz_list, L)
        plotImba.append(imba)
        print(f"Imbalance at time t={t}: {imba}")

    axs[0].plot(time, plotE, label=f"W={W}")
    axs[1].plot(time, plotImba, label=f"W={W}")

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    axs[0].set_xlabel('Time', fontsize=14)
    axs[0].set_ylabel('Half-chain Entanglement Entropy', fontsize=14)
    axs[1].set_xlabel('Time', fontsize=14)
    axs[1].set_ylabel('Imbalance', fontsize=14)

    axs[0].legend(fontsize=14)
    axs[1].legend(fontsize=14)

    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Bplot_Entanglement_Imbalance_w5_{current_time}.jpg"
    plt.savefig(filename)
    plt.show()