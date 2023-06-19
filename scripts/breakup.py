import numpy as np
import pandas as pd
import glob

def mass_from_log_g(log_g, R):
    G = 6.67430e-11  # m^3 kg^-1 s^-2 (gravitational constant)
    M_sun = 1.9885e30  # kg (solar mass)
    R_sun = 6.9634e8   # m (solar radius)

    R_m = R * R_sun
    g = 10 ** log_g

    M_kg = (g * R_m ** 2) / G
    M = M_kg / M_sun

    return M

def breakup_velocity(M, R):
    G = 6.67430e-11  # m^3 kg^-1 s^-2 (gravitational constant)
    M_sun = 1.9885e30  # kg (solar mass)
    R_sun = 6.9634e8   # m (solar radius)

    M_kg = M * M_sun
    R_m = R * R_sun

    v = np.sqrt(G * M_kg / R_m)

    return v / 1000  # Convert to km/s

if __name__ == "__main__":
    M = 1.22
    log_g = 3.9348807733425635
    R = 10**0.2947733423176589
    M = mass_from_log_g(log_g, R)

    print(breakup_velocity(M, R))