def _do_iterative(S, PHI, Lin, Lout, ni=5):
    """Compute SSS weights using iterative method.
    
    Code for iterative implementation of the Signal Space Separation method.
    Based on 'An iterative implementation of the signal space separation method 
    for magnetoencephalography systems with low channel counts' by Niall Holmes,
    Richard Bowtell, Matthew Brooked and Samu Taulu.
    
    The method is based on the assumption that the SSS vectors represent the
    MEG data in a hierarchical manner where the low-order components always
    explain a larger amount of signal energy than the high-order components.
    
    Parameters
    ----------
    S : ndarray, shape (n_channels, n_basis)
        Column normalized SSS basis.
    PHI : ndarray, shape (n_channels, n_times)
        MEG data.
    Lin : int
        Number of inner harmonics used in S.
    Lout : int
        Number of outer harmonics used in S.
    ni : int
        Number of iterations (default: 5).
        
    Returns
    -------
    X : ndarray, shape (n_basis, n_times)
        SSS weights found via iterative method. Estimate signal as S @ X.
    """
    nsamp = PHI.shape[1]  # Number of time samples
    dim_m = (Lin + 1) ** 2 - 1  # Dimension of the internal SSS basis
    
    # Extract the column vectors corresponding to each l-value of the internal basis
    dimv = []
    for n in range(1, Lin + 1):
        dim1 = (n - 1 + 1) ** 2
        dim2 = (n + 1) ** 2 - 1
        dimv.append([dim1 - 1, dim2 - 1])  # Convert to 0-based indexing
    
    X = np.zeros((S.shape[1], nsamp))  # Initial zero weights vector
    
    # Pre-compute indices and pseudoinverses for each order
    indices = []
    pS = []
    for n in range(Lin):
        # Indices for Lin-specific components and all Lout components
        idx = list(range(dimv[n][0], dimv[n][1] + 1)) + list(range(dim_m, dim_m + (Lout + 1) ** 2 - 1))
        indices.append(idx)
        # Pre-computed pseudoinverse matrices for individual l-values
        pS.append(linalg.pinv(S[:, idx]))
    
    if Lin >= Lout:  # Check dimensions okay
        for i in range(ni):  # For each iteration
            for j in range(Lin):  # For each inner order
                inds = indices[j]  # Find relevant indices
                X[inds, :] = 0  # Zero relevant weights
                XN = pS[j] @ (PHI - S @ X)  # Update the l-specific multipole moments
                X[inds, :] = XN  # Update weights
    else:
        raise ValueError('Lin should be at least as large as Lout!')
    
    return X