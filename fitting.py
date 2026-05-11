import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import warnings

# Enable 64-bit precision for accurate Hessian and Covariance calculations
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)

def fit_model_jax(data, cov_matrix_or_errors, fitting_function, initial_params, 
                  start=0, end=None, method='BFGS', priors=0.0, 
                  max_cond_number=1e15):
    """
    Fits the data using JAX automatic differentiation for exact gradients and Hessians.
    
    data: list of [x, y] or [x, y, error]
    cov_matrix_or_errors: 2D covariance matrix or 1D error vector.
    fitting_function: python function f(x, params) using ONLY jax.numpy operations.
    initial_params: 1D array of initial guesses for parameters.
    """
    data = np.array(data)
    
    # Handle [x, y, err] format
    if data.shape[1] == 3:
        cov_matrix_or_errors = data[:, 2]
        data = data[:, :2]
        
    if end is None:
        end = len(data)
        
    x_data = jnp.array(data[start:end, 0])
    y_data = jnp.array(data[start:end, 1])
    
    cov_matrix_or_errors = jnp.array(cov_matrix_or_errors)
    if cov_matrix_or_errors.ndim == 1:
        reduce_cov_matrix = jnp.diag(cov_matrix_or_errors[start:end]**2)
    else:
        reduce_cov_matrix = cov_matrix_or_errors[start:end, start:end]
        
    num_params = len(initial_params)
    initial_params = jnp.array(initial_params)
    
    # Check the conditioning number for the covariance matrix
    sigma_list = jnp.sqrt(jnp.diag(reduce_cov_matrix))
    norm_matrix = reduce_cov_matrix / jnp.outer(sigma_list, sigma_list)
    norm_matrix_eigenval = jnp.linalg.eigvals(norm_matrix)
    
    cond_number = jnp.max(jnp.abs(norm_matrix_eigenval)) / jnp.min(jnp.abs(norm_matrix_eigenval))
    
    if cond_number > max_cond_number or jnp.min(norm_matrix_eigenval) < 0:
        val, rot = jnp.linalg.eigh(norm_matrix)
        eigenval_threshold = 1.0 / max_cond_number
        
        # Invert small eigenvalues safely
        inv_val = jnp.where(jnp.abs(val) < eigenval_threshold, 1.0 / eigenval_threshold, 1.0 / val)
        inv_cov_matrix = rot @ jnp.diag(inv_val) @ jnp.conjugate(rot).T
        inv_cov_matrix = inv_cov_matrix / jnp.outer(sigma_list, sigma_list)
    else:
        inv_cov_matrix = jnp.linalg.inv(reduce_cov_matrix)
        
    # Ensure symmetry
    inv_cov_matrix = (inv_cov_matrix + inv_cov_matrix.T) / 2.0
    
    # Define vectorized model prediction for fast execution
    @jax.jit
    def model_preds(params):
        # Maps the fitting_function over x_data, keeping params constant
        return jax.vmap(fitting_function, in_axes=(0, None))(x_data, params)

    # Define Chi-square objective function
    @jax.jit
    def chi2(params):
        residuals = model_preds(params) - y_data
        return residuals.T @ inv_cov_matrix @ residuals + priors

    # Compile function that returns both objective value and gradient
    chi2_val_and_grad = jax.jit(jax.value_and_grad(chi2))
    
    # Wrapper for SciPy minimize (SciPy expects float64 numpy arrays, not JAX arrays)
    def scipy_objective(p):
        val, grad = chi2_val_and_grad(jnp.array(p))
        return np.array(val), np.array(grad)

    # Perform minimization using SciPy with exact JAX gradients
    solution = opt.minimize(scipy_objective, np.array(initial_params), method=method, jac=True)
    
    best_params = jnp.array(solution.x)
    final_chi2 = solution.fun
    
    # Calculate exact Hessian and Jacobian at the solution using JAX autodiff
    hessian_func = jax.jit(jax.hessian(chi2))
    jacobian_func = jax.jit(jax.jacfwd(model_preds))
    
    hessian = hessian_func(best_params)
    # Transpose Jacobian to match Mathematica's shape: (num_params, num_data)
    deriv = jacobian_func(best_params).T 
    
    # Regularize Hessian eigenvalues
    val, rot = jnp.linalg.eigh(hessian)
    max_v = jnp.max(val)
    inv_val = jnp.where(val < max_v * 1e-15, 1e15 / max_v, 1.0 / val)
    inv_hessian = rot @ jnp.diag(inv_val) @ jnp.conjugate(rot).T
    
    # Compute Parameter Covariance (Delta)
    delta = 4 * inv_hessian @ deriv @ inv_cov_matrix @ deriv.T @ inv_hessian
    errors = jnp.sqrt(jnp.diag(jnp.abs(delta)))
    
    dof = len(x_data) - num_params
    
    # Convert JAX arrays back to standard NumPy arrays for the return dictionary
    return {
        'Solution': np.array(best_params),
        'SimpleOutput': np.column_stack((np.array(best_params), np.array(errors))),
        'HessianMat': np.array(hessian),
        'Delta': np.array(delta),
        'ChiSquareFunction': float(final_chi2),
        'DegreesOfFreedom': dof
    }

def print_fit_report(fitres):
    """
    Prints a formatted report of the fitting results.
    """
    chisq = fitres['ChiSquareFunction']
    dof = fitres['DegreesOfFreedom']
    
    # Calculate confidence level
    conflev = 1 - stats.chi2.cdf(chisq, dof)
    ds = np.sqrt(np.diag(np.abs(fitres['Delta'])))
    
    num_params = len(ds)
    ndelta = np.zeros((num_params, num_params))
    for r in range(num_params):
        for c in range(num_params):
            if r <= c:
                ndelta[r, c] = fitres['Delta'][r, c] / (ds[r] * ds[c])
            else:
                ndelta[r, c] = np.nan
                
    print("Hessian=\n", fitres['HessianMat'])
    print("\nDelta=\n", fitres['Delta'])
    print("\nCovMat (Correlation)=\n", ndelta)
    
    print(f"\nChi^2 = {chisq}    Chi^2/dof = {chisq/dof}")
    print(f"Confidence level = {conflev}")
    
    print("\nFinal Results")
    print(f"{'Parameter':>10} | {'Best fit':>15} | {'Error':>15}")
    print("-" * 46)
    for i, (param, error) in enumerate(fitres['SimpleOutput']):
        print(f"{'p'+str(i):>10} | {param:>15.6f} | {error:>15.6f}")