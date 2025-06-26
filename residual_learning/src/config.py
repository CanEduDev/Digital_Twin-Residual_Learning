# src/config.py
import math

# --- BICYCLE PARAMETERS ---
L         = 1.0       # wheel-base [m]
K_DELTA   = 1.38      # steering gain (column → road-wheel)
TAU_DELTA = 0.028     # steering first-order lag [s]
CD        = 3.0e-5    # quadratic drag [s/m]

# --- SIMULATION CONSTANTS ---
MAX_STEER_RAD = math.pi/2 - 1e-3