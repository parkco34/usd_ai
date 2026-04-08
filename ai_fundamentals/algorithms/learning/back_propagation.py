#!/usr/bin/env python
"""
ALGORITHM
---------------------------------------------------------
BACKPROP_NETWORK(X, y, architecture, η, epochs)

    1. INITIALIZE weights and biases for each layer
    2. FOR each epoch:
        a. FORWARD PASS:
           For l = 1 to L:
               z[l] = W[l] @ a[l-1] + b[l]
               a[l] = g(z[l])
           where a[0] = X

        b. COMPUTE LOSS:
           L = -(1/m) * sum( y*ln(a[L]) + (1-y)*ln(1-a[L]) )

        c. BACKWARD PASS:
           δ[L] = a[L] - y
           For l = L-1 down to 1:
               δ[l] = (W[l+1]ᵀ @ δ[l+1]) ⊙ g'(z[l])

        d. COMPUTE GRADIENTS:
           For l = 1 to L:
               dW[l] = (1/m) * δ[l] @ a[l-1]ᵀ
               db[l] = (1/m) * sum(δ[l], across samples)

        e. UPDATE PARAMETERS:
           For l = 1 to L:
               W[l] = W[l] - η * dW[l]
               b[l] = b[l] - η * db[l]

    3. RETURN trained W, b
--------------------------------------------------------
STEPS
1.
"""

