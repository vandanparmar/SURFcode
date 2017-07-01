#Playing around with some linear algebra functions

import numpy as np
import numpy.linalg as la



n = 4;
I4 = np.eye(n)
A = np.random.randn(n,n)

U,s,V = la.svd(A, full_matrices = False)

print("Matrix we start with")
print(A)
A = np.matrix(A)
print("Transpose")
print(A.H)

S = np.diag(s)
A_recon = np.matrix(U)*np.matrix(S)*np.matrix(V)
print("Compute SVD of A and reconstruct")
print(A_recon)
err = A-A_recon

print("Eror between A and SVD reconstruction")
print(err)
print("Frobenius norm of error")
print(la.norm(err,'fro'))


