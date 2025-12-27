from libintrinsic.datasets.loaders import get_sphere


def main():
    print("--- Testing LibIntrinsic Mesh Loading ---")

    # 1. Load the test shape
    try:
        mesh = get_sphere(subdivisions=3)
        print(f"Success: Loaded {mesh}")
        print(f"   - Vertices: {mesh.vertices.shape}")
        print(f"   - Faces:    {mesh.faces.shape}")

    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    main()