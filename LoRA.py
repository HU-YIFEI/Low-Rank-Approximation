import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv, eig
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm


def opt_cssp(A: np.ndarray, k: int):
    """
    Compute OPT solution for CSSP, given the solution in terms of Frobenius
    norm and spectral norm, as well as the optimal rank k approximation of A
    in Frobenius norm. Args: A (np.ndarray): the input matrix to be
    approximated k (int): the rank of the solution

    Returns: the optimal rank k approximation of A, the optimal solution in
    terms of Frobenius norm, and the optimal solution in terms of spectral
    norm.
    """
    C = A
    Q, _ = np.linalg.qr(C)
    QTA = Q.T @ A
    U, S, VT = np.linalg.svd(QTA)
    sol = Q @ U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
    fro = np.linalg.norm(A - sol, ord='fro')
    spec = np.linalg.norm(A - sol, ord=2)
    return sol, fro, spec


def gram_schmidt(V: np.ndarray):
    """
    Apply the Gram-Schmidt process to orthonormalize a set of vectors.

    Args:
        V (np.ndarray): The matrix to be orthonormalized.

    Returns:
        np.ndarray: An orthogonal basis of the span of V.
    """
    if V.ndim == 1:
        raise ValueError("Expected a 2D array, got a 1D array instead.")
    U = np.zeros_like(V).astype(np.float64)
    for i in range(V.shape[0]):
        U[i] = V[i]
        for j in range(i):
            U[i] -= np.dot(U[j], V[i]) * U[j]
        
        norm_U_i = np.linalg.norm(U[i])
        if norm_U_i > 1e-10:  # Threshold to avoid division by near zero
            U[i] /= norm_U_i
        else:
            U[i] = np.zeros_like(U[i])  # Handle the zero vector case
    return U


def project_matrix_onto_span(A, S):
    """
    Project each row of matrix A onto a linear subspace spanned by the rows
    in S.

    Args:
    A (np.ndarray): The original matrix.
    S (np.ndarray): A subset of rows that span the subspace.
    k (int): The rank for the approximation.

    Returns:
    np.ndarray: The rank-k approximation of A with rows in the span of S.
    """
    # Perform SVD on the subset of rows S
    U, Sigma, VT = np.linalg.svd(S, full_matrices=False)
    VT_k = VT[:, :]
    # Project A onto the span of S_k
    A_projected = np.dot(A, np.dot(VT_k.T, VT_k))
    return A_projected


def adaptive_sampling(A: np.ndarray, k: int):
    """
    Use the adaptive sampling algorithm to select k < m rows from A.
    Algorithm follows DESHPANDE, A., AND VEMPALA, S. Adaptive sampling and
    fast low-rank matrix approximation. In Approximation, Randomization,
    and Combinatorial Optimization. Algorithms and Techniques (Berlin,
    Heidelberg, 2006), J. Díaz, K. Jansen, J. D. P. Rolim, and U. Zwick,
    Eds., Springer Berlin Heidelberg, pp. 292–303

    Args:
        A (np.ndarray): the input matrix to be approximated
        k (int): the rank of the solution
    """
    E = A
    k = int(k)
    S = np.zeros((k, A.shape[1]))
    for i in range(k):
        p = np.sum(np.square(E), axis=1)
        p = p / np.sum(p)
        idx = np.random.choice(A.shape[0], p=p)
        S[i, :] = A[idx, :]
        E = A - project_matrix_onto_span(A, S)
    return S


def adaptive_sampling_wp(A: np.ndarray, P: np.ndarray, k: int):
    """
    Use the adaptive sampling algorithm to select k < m rows from A.
    Algorithm follows DESHPANDE, A., AND VEMPALA, S. Adaptive sampling and
    fast low-rank matrix approximation. In Approxima- tion, Randomization,
    and Combinatorial Optimization. Algorithms and Techniques (Berlin,
    Heidelberg, 2006), J. Díaz, K. Jansen, J. D. P. Rolim, and U. Zwick,
    Eds., Springer Berlin Heidelberg, pp. 292–303

    Args: A (np.ndarray): the input matrix to be sampled from P (
    np.ndarray): the matrix used to calculate the sampling probability,
    should have the same number of rows as A k (int): the rank of the solution
    """
    E = P
    k = int(k)
    S = np.zeros((k, A.shape[1]))
    for i in range(k):
        p = np.sum(np.square(E), axis=1)
        p = p / np.sum(p)
        idx = np.random.choice(A.shape[0], p=p)
        S[i, :] = A[idx, :]
        E = A - project_matrix_onto_span(A, S)
    return S


def dual_set_spectral_sparsification(V, U, r):
    """
    Implement the Deterministic Dual Set Spectral Sparsification algorithm.
    """
    
    def phi_lo(L, A):
        """
        Compute the sum of the inverse differences between eigenvalues of A
        and L.
        """
        eigenvalues = eig(A)[0]
        return sum([1 / (lam - L) for lam in eigenvalues])
    
    def phi_up(U, B):
        """
        Compute the sum of the inverse differences between U and eigenvalues
        of B.
        """
        eigenvalues = eig(B)[0]
        return sum([1 / (U - lam) for lam in eigenvalues])
    
    def L_function(v, delta_L, A, L):
        """
        Compute the L function as specified.
        """
        term1 = v.T @ inv(A - (L + delta_L) * np.eye(A.shape[0])) @ v
        term2 = v.T @ inv(A - (L + delta_L) * np.eye(A.shape[0])) ** 2 @ v
        term1 = np.real(term1)
        term2 = np.real(term2)
        a = np.real(phi_lo(L + delta_L, A))
        b = np.real(phi_lo(L, A))
        term2 /= (a - b)
        return -term1 + term2
    
    def U_function(u, delta_U, B, U):
        """
        Compute the U function as specified.
        """
        term1 = u.T @ inv((U + delta_U) * np.eye(B.shape[0]) - B) @ u
        term2 = u.T @ inv((U + delta_U) * np.eye(B.shape[0]) - B) ** 2 @ u
        a = np.real(phi_up(U, B))
        b = np.real(phi_up(U + delta_U, B))
        term2 = np.real(term2)
        term2 /= a - b
        return term1 + term2
    
    n, k = V.shape[1], V.shape[0]
    l = U.shape[0]
    delta_U = (1 + np.sqrt(l / r)) ** 2 / (
            1 - np.sqrt(k / r))  # Follows from the paper
    delta_L = 1  # Follows from the paper
    
    s = np.zeros(n)
    A = np.zeros((k, k))
    B = np.zeros((l, l))
    
    for tau in range(r):
        L_tau = tau - np.sqrt(r * k)
        U_tau = delta_U * (tau + np.sqrt(l * r))
        # Find the index j according to the condition in step 2
        j = None
        for i in range(n):
            v_j = V[:, i:i + 1]
            u_j = U[:, i:i + 1]
            u = U_function(u_j, delta_U, B, U_tau)
            L = L_function(v_j, delta_L, A, L_tau)
            if u[0, 0] <= L[0, 0]:
                j = i
                break
        if j is None:
            j = 0
        
        t_inv = 0.5 * (
                U_function(U[:, j:j + 1], delta_U, B, U_tau) + L_function(
            V[:, j:j + 1], delta_U, A, L_tau))
        t = 1 / t_inv
        
        # Update s, A, and B
        s[j] += t
        A += t * np.outer(V[:, j], V[:, j])
        B += t * np.outer(U[:, j], U[:, j])
    
    return (1 - np.sqrt(k / r)) * s


def near_optimal_CSSP_Boutsidis_2011(A: np.ndarray, k: int, eps: float,
                                     c: int):
    """
    Compute near optimal solution for CSSP, given the solution in terms of
    Frobenius norm and spectral norm, as well as the optimal rank k
    approximation of A in Frobenius norm. Args: A (np.ndarray): the input
    matrix to be approximated k (int): the rank of the solution eps (float):
    the error parameter c (int): the number of target columns number


    Returns: the near optimal rank k approximation of A, the near optimal
    solution in terms of Frobenius norm, and the near optimal solution in
    terms of spectral norm.
    """
    # Step 1: Input the matrix A, the target rank k, and the error parameter
    # epsilon - trivial Step 2: Compute the full SVD of A
    tsvd = TruncatedSVD(n_components=k,
                        algorithm='randomized')  # note that the
    # approximation ratio depends also on the hyperparameter choice for the
    # randomized SVD algorithm, with algorithm='randomized', the code used
    # the exact algorimthm by Halko et al. 2009 as in the paper
    tsvd.fit(A)
    VT = tsvd.components_
    sig = tsvd.singular_values_
    USig = tsvd.transform(A)
    U = USig.dot(np.linalg.inv(np.diag(sig)))
    # Step 3: Construct U and V from the SVD of A
    U = A - U @ np.diag(sig) @ VT
    
    num_dual = int(c - np.ceil(2 * k / eps))
    num_dual = max(num_dual, VT.shape[0])
    num_dual = min(num_dual, U.shape[1] - 1)
    # Step 4: dual set spectral Frobenius sparsification
    s = dual_set_spectral_sparsification(U, VT, num_dual)
    
    # Step 5
    C1 = A @ np.diag(s)
    C1 = C1[:, ~np.all(C1 == 0, axis=0)]
    res = A - C1 @ np.linalg.pinv(C1) @ A
    num_c2 = c - C1.shape[1]
    num_c2 = min(num_c2, res.shape[1])
    C2 = adaptive_sampling(res.T, num_c2)
    C2 = C2.T
    C = np.concatenate((C1, C2), axis=1)
    approx = C @ np.linalg.pinv(C) @ A
    fro = np.linalg.norm(A - approx, ord='fro')
    spec = np.linalg.norm(A - approx, ord=2)
    return approx, fro, spec, C


def adaptive_CUR(A: np.ndarray, k: int, c: int, eps: float):
    """
    Compute near optimal solution for CSSP, given the solution in terms of
    Frobenius norm and spectral norm, as well as the optimal rank k
    approximation of A in Frobenius norm. Algorithm follows S. Wang and Z.
    Zhang, Improving CUR matrix decomposition and the Nystro ̈m approximation
    via adaptive sampling, Journal of Machine Learning Research, 14, 2549-2589,
     2013.
    Args:
        A (np.ndarray): the input matrix to be approximated
        k (int): the rank of the solution
        c (int): the number of target columns number
        eps (float): the error parameter
    """
    num_col = int(np.floor(2 * k / eps))
    num_row = int(np.floor(c / eps))
    num_row = min(num_row, A.shape[0])
    _, _, _, C = near_optimal_CSSP_Boutsidis_2011(A, k, eps=eps, c=num_col)
    _, _, _, R1 = near_optimal_CSSP_Boutsidis_2011(A.T, k, eps, num_col)
    R2 = adaptive_sampling_wp(A, A - np.dot(A, np.dot(R1, np.linalg.pinv(R1))),
                              num_row)
    R2 = R2.T
    R = np.concatenate((R1, R2), axis=1)
    R = R.T
    U = np.dot(np.linalg.pinv(C), np.dot(A, np.linalg.pinv(R)))
    approx = np.dot(C, np.dot(U, R))
    frob = np.linalg.norm(A - approx, ord='fro')
    spec = np.linalg.norm(A - approx, ord=2)
    return C, U, R, approx, frob, spec


def randomized_svd(A, l):
    """
    Perform Randomized Singular Value Decomposition (SVD) on a matrix A.

    Args:
        A (numpy.ndarray): Input matrix of size m x n.
        l (int): The number of random projections.

    Returns:
        U (numpy.ndarray): Left singular vectors.
        S (numpy.ndarray): Singular values.
        VT (numpy.ndarray): Right singular vectors (transposed).

    Raises:
        ValueError: If l is greater than min(m, n).
    """
    m, n = A.shape
    if l > min(m, n):
        raise ValueError(
            "The number of random projections 'l' should be less "
            "than min(m, n).")
    
    # Stage 1: Randomized Range Finder
    Omega = np.random.normal(size=(n, l))
    Y = A.dot(Omega)
    Q, _ = np.linalg.qr(Y)  # QR factorization
    
    # Stage 2: Direct SVD
    B = Q.T.dot(A)
    U_tilde, Sigma, VT = np.linalg.svd(B, full_matrices=False)
    U = Q.dot(U_tilde)
    approx = U.dot(np.diag(Sigma)).dot(VT)
    fro = np.linalg.norm(A - approx, ord='fro')
    spec = np.linalg.norm(A - approx, ord=2)
    return approx, fro, spec


def randomized_projection(A: np.ndarray, k: int, num_it: int):
    """
    Perform Randomized projection on a matrix A to find its low rank
    approximation. Args: A (): matrix to be approximated k (): target rank
    num_it (): number of subspace iterations

    Returns: Q matrix of the low rank approximation

    """
    omega = np.random.normal(size=(A.shape[1], k))
    Y = A @ omega
    Q, _ = np.linalg.qr(Y)
    for i in range(num_it):
        Y = A.T @ Q
        Q, _ = np.linalg.qr(Y)
        Y = A @ Q
        Q, _ = np.linalg.qr(Y)
    
    approx = Q @ Q.T @ A
    fro = np.linalg.norm(A - approx, ord='fro')
    spec = np.linalg.norm(A - approx, ord=2)
    return approx, fro, spec


def matrix_approximation_experiment(matrix_sizes, distributions,
                                    num_iterations, k, c, eps):
    """
    Perform an experiment to compare matrix approximation methods.

    Args: matrix_sizes (list of tuples): List of matrix sizes as (rows,
    columns). distributions (list of str): List of distributions to generate
    matrices ('uniform', 'normal', 'exponential'). num_iterations (int):
    Number of iterations for averaging results. k (int): Target rank for
    approximations. c (int): Number of columns/rows for randomized CUR and
    near optimal CSSP.
        
        
        eps (float): Error parameter for approximations.

    Returns:
        dict: A dictionary containing the averaged results of the experiment.
    """
    results = {size: {dist: {'near_optimal_CSSP': [], 'adaptive_CUR': [],
                             'randomized_svd': [], 'randomized_projection': []}
                      for dist in distributions} for size in matrix_sizes}
    
    for size in tqdm(matrix_sizes, desc='Matrix Size', position=0, leave=True):
        for dist in distributions:
            for _ in tqdm(range(num_iterations),
                          desc=f'Iterations for {size},{dist}', position=0,
                          leave=True):
                # Generate random matrix
                if dist == 'uniform':
                    A = np.random.uniform(low=-1, high=1, size=size)
                elif dist == 'normal':
                    A = np.random.normal(loc=0, scale=1, size=size)
                elif dist == 'exponential':
                    A = np.random.exponential(scale=1, size=size)
                else:
                    raise ValueError(f'Invalid distribution: {dist}')
                
                # Compute the optimal solution
                opt_sol, opt_fro, opt_spec = opt_cssp(A, k)
                
                # Apply different approximation methods
                approx1, fro1, spec1, _ = near_optimal_CSSP_Boutsidis_2011(A,
                                                                           k,
                                                                           eps,
                                                                           c)
                _, _, _, approx2, fro2, spec2 = adaptive_CUR(A, k, c, eps)
                approx3, fro3, spec3 = randomized_svd(A, k)
                approx4, fro4, spec4 = randomized_projection(A, k, 3)
                # Assuming 10 iterations for subspace iterations
                
                # Store the results: the 2 factor is to account for the fact
                # that svd is 2-approximation for cssp-s problem.
                results[size][dist]['near_optimal_CSSP'].append(
                    (fro1 / opt_fro, 2 * spec1 / opt_spec))
                results[size][dist]['adaptive_CUR'].append(
                    (fro2 / opt_fro, 2 * spec2 / opt_spec))
                results[size][dist]['randomized_svd'].append(
                    (fro3 / opt_fro, 2 * spec3 / opt_spec))
                results[size][dist]['randomized_projection'].append(
                    (fro4 / opt_fro, 2 * spec4 / opt_spec))
    
    # Compute average results
    average_results = {size: {dist: {} for dist in distributions} for size in
                       matrix_sizes}
    for size in matrix_sizes:
        for dist in distributions:
            for method in results[size][dist]:
                fro_errors = [error[0] for error in
                              results[size][dist][method]]
                spec_errors = [error[1] for error in
                               results[size][dist][method]]
                average_results[size][dist][method] = (
                    np.mean(fro_errors), np.mean(spec_errors))
    
    return average_results


def generate_results_table(average_results):
    """
    Generate a Pandas DataFrame from the average results of the experiment.

    Args:
        average_results (dict): The averaged results from the matrix
        approximation experiment.

    Returns:
        pd.DataFrame: A DataFrame representing the results in tabular format.
    """
    data = []
    for size in average_results:
        for dist in average_results[size]:
            for method in average_results[size][dist]:
                fro_error, spec_error = average_results[size][dist][method]
                data.append([size, dist, method, fro_error, spec_error])
    
    df = pd.DataFrame(data, columns=['Size', 'Distribution', 'Method',
                                     'Avg Frobenius Error Ratio',
                                     'Avg Spectral Error Ratio'])
    return df


def plot_results_graph(average_results):
    """
    Plot the results of the matrix approximation experiment.
    Args:
        average_results (dict): The averaged results from the matrix
        approximation experiment.
    """
    for size in average_results:
        plt.figure(figsize=(10, 15))
        for dist in average_results[size]:
            methods = list(average_results[size][dist].keys())
            fro_errors = [average_results[size][dist][method][0] for method in
                          methods]
            spec_errors = [average_results[size][dist][method][1] for method in
                           methods]
            
            plt.plot(methods, fro_errors, label=f'{dist} Frobenius',
                     marker='o')
            plt.plot(methods, spec_errors, label=f'{dist} Spectral',
                     marker='x')
        
        plt.title(f'Matrix Approximation Results for Size {size}')
        plt.xlabel('Method')
        plt.ylabel('Approximation Ratio')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.show()
