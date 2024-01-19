import numpy as np
import matplotlib.pyplot as plt

# get epsilon from n
# n^2 * (pi^2 / g2) - 1
# solve to get epsilon


N = 10000 # number of iterations
Psi = np.zeros(N) # wave function, start as empty array
g2 = 200 # gamma squared
v = (-1)*np.ones(N) # potential. array
ep = 3.93 * np.ones(N) # triat energy
k2 = g2*(ep - v) #k squared
l2 = (1.0/(N-1))**2 #l squared
def wavefunction(ep, N) : # finds wavefunction
    Psi[0] = 0 # set first two values of psi
    Psi[1] = 1e-4

    for i in range(2, N):
        Psi[i] = (2*(1-(5.0/12)*l2*k2[i-1])*Psi[i-1] -
                  (1+(1.0/12)*l2*k2[i-2])*Psi[i-2])/(1+(1.0/12)*l2*k2[i])

    return Psi
res = wavefunction(ep,N)

x_values = np.linspace(0, 1, N)  # Assuming a spatial domain from 0 to 1

# Plotting the wave function
plt.figure(figsize=(8, 6))
plt.plot(x_values, res, label='Wave Function')
plt.legend()
plt.grid(True)
plt.show()



