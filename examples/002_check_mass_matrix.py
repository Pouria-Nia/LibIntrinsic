import numpy as np

from libintrinsic.datasets.loaders import get_sphere
from libintrinsic.spectral.laplacian import compute_mass_matrix


def main():
    # 1. Load Sphere (r=1.0)
    mesh= get_sphere(radius=1.0, subdivisions=4)
    print(f"Loaded Mesh: {mesh}")

    # 2. Compute Mass Matrix
    M = compute_mass_matrix(mesh)
    print(f"Mass Matrix shape: {M.shape}")

    # 3. Physical Verification
    # The sum of all diagonal elements should be the total surface area
    calculated_area = M.sum()
    theoretical_area = 4 * np.pi * (1.0 ** 2)

    print(f"Calculated Area:  {calculated_area:.5f}")
    print(f"Theoretical Area: {theoretical_area:.5f}")

    error = theoretical_area - calculated_area
    print(f"Difference:       {error:.5f}")

    if error < 0.1:
        print("✅ Physics Check Passed: Mass Matrix represents surface area.")
    else:
        print("❌ Physics Check Failed.")


if __name__ == "__main__":
    main()