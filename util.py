import numpy as np
import jax
import jax.numpy as jnp
import flax


def jackknife(xs, ws=None, Bs=50):  # Bs: Block size
    B = len(xs)//Bs  # number of blocks
    if ws is None:  # for reweighting
        ws = xs*0 + 1

    x = np.array(xs[:B*Bs])
    w = np.array(ws[:B*Bs])

    m = sum(x*w)/sum(w)

    # partition
    block = [[B, Bs], list(x.shape[1:])]
    block_f = [i for sublist in block for i in sublist]
    x = x.reshape(block_f)
    w = w.reshape(block_f)

    # jackknife
    vals = [np.mean(np.delete(x, i, axis=0)*np.delete(w, i, axis=0)) /
            np.mean(np.delete(w, i)) for i in range(B)]
    vals = np.array(vals)
    return m, (np.std(vals.real) + 1j*np.std(vals.imag))*np.sqrt(len(vals)-1)


def jackknife_effmass(xs, Bs=50):  # Bs: Block size
    N_conf, T = xs.shape

    B = len(xs)//Bs  # number of blocks

    data = xs[:B*Bs]

    blocks = data.reshape(B, Bs, T)
    block_means = np.mean(blocks, axis=1)  # shape: (B, T)

    total_sum = np.sum(block_means, axis=0)
    jack_samples = (total_sum-block_means) / (B-1)

    # Effective mass
    eff_mass_jack = np.log(jack_samples[:, :-1] / jack_samples[:, 1:])

    # mean
    total_mean_correlator = np.mean(data, axis=0)
    eff_mass_means = np.log(
        total_mean_correlator[:-1] / total_mean_correlator[1:])

    # jackknife errors
    errors = np.sqrt(B-1) * np.std(eff_mass_jack, axis=0)

    return eff_mass_means, errors


def bin(xs, ws=None, Bs=50):  # Bs: Block size
    B = len(xs)//Bs  # number of blocks
    if ws is None:  # for reweighting
        ws = xs*0 + 1

    x = np.array(xs[:B*Bs])
    w = np.array(ws[:B*Bs])

    m = sum(x*w)/sum(w)

    # partition
    block = [[B, Bs], list(x.shape[1:])]
    block_f = [i for sublist in block for i in sublist]
    x = x.reshape(block_f)
    w = w.reshape(block_f)

    # jackknife
    vals = [np.mean(x[i]*w[i])/np.mean(w[i]) for i in range(B)]
    vals = np.array(vals)
    return m, (np.std(vals.real) + 1j*np.std(vals.imag))/np.sqrt(len(vals)-1)


def bootstrap(xs, ws=None, N=100, Bs=50):
    if Bs > len(xs):
        Bs = len(xs)
    B = len(xs)//Bs
    if ws is None:
        ws = xs*0 + 1
    # Block
    x, w = [], []
    for i in range(Bs):
        x.append(sum(xs[i*B:i*B+B]*ws[i*B:i*B+B])/sum(ws[i*B:i*B+B]))
        w.append(sum(ws[i*B:i*B+B]))
    x = np.array(x)
    w = np.array(w)
    # Regular bootstrap
    y = x * w
    m = (sum(y) / sum(w))
    ms = []
    for n in range(N):
        s = np.random.choice(range(len(x)), len(x))
        ms.append((sum(y[s]) / sum(w[s])))
    ms = np.array(ms)
    return m, np.std(ms.real) + 1j*np.std(ms.imag)


# regularizations
def l2_loss(x, alpha):
    return alpha*(x**2).mean()


def l1_loss(x, alpha):
    return alpha*(abs(x)).mean()


def l2_regularization(params):
    # Flatten the nested parameter dict.
    flat_params = flax.traverse_util.flatten_dict(params)
    # Sum up the L2 norm of all parameters where the key ends with 'kernel'
    l2_sum = sum(jnp.sum(param ** 2)
                 for key, param in flat_params.items() if key[-1] == 'kernel')
    return l2_sum


def l1_regularization(params):
    # Flatten the nested parameter dict.
    flat_params = flax.traverse_util.flatten_dict(params)
    # Sum up the L2 norm of all parameters where the key ends with 'kernel'
    l2_sum = sum(jnp.sum(jnp.abs(param))
                 for key, param in flat_params.items() if key[-1] == 'kernel')
    return l2_sum


def decay_mask(params):
    # For adamW
    ''' 
    Example: optax.adamw(learning_rate=1e-3, weight_decay=1e-4, mask=decay_mask(params))
    '''
    flat = flax.traverse_util.flatten_dict(params)
    mask = {path: (path[-1] == "kernel") for path in flat}
    return flax.traverse_util.unflatten_dict(mask)


def autocorr_time_fft(x, max_lag=None):
    x = np.asarray(x)
    n = len(x)
    x = x - np.mean(x)
    var = np.var(x, ddof=1)
    if var == 0:
        return 1.0  # or np.nan

    # Next power of 2 for efficient FFT
    n_fft = 2 ** (int(np.ceil(np.log2(2 * n))))
    # print(n_fft)

    # FFT of zero-padded signal
    fx = np.fft.fft(x, n=n_fft)
    acf = np.fft.ifft(fx * np.conjugate(fx)).real[:n]

    # Normalize
    acf /= var * np.arange(n, 0, -1)
    acf /= acf[0]

    # Set max lag
    if max_lag is None:
        max_lag = n // 2
    max_lag = min(max_lag, len(acf) - 1)

    # Initial positive sequence
    t = 1
    while t < max_lag and acf[t] > 0:
        t += 1

    tau = 1 + 2 * np.sum(acf[1:t])
    return tau
