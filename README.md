# LibIntrinsic
A Python library for intrinsic shape analysis, spectral geometry, and topological descriptors. Define geometry by its structure, not its coordinates.



libintrinsic/
│
├── libintrinsic/               # Actual Python package
│   ├── __init__.py
│   │
│   ├── spectral/               # RENAMED from 'lbo'
│   │   ├── __init__.py
│   │   ├── laplacian.py        # Logic to build the LBO Matrix (Cotangent)
│   │   ├── descriptors.py      # HKS, WKS, GPS (The "Signatures")
│   │   └── utils.py            # Helpers specific to spectral stuff
│   │
│   ├── geometry/               # Mesh data structures
│   │   ├── __init__.py
│   │   ├── mesh.py             # A class to hold (V, F) data
│   │   └── io.py               # Load/Save .obj files
│   │
│   ├── numerics/               # The Math Engine
│   │   ├── __init__.py
│   │   ├── solvers.py          # Eigendecomposition wrapper (sparse/dense)
│   │   └── linear.py           # Matrix multiplication helpers
│   │
│   ├── datasets/               # Test data
│   │   ├── __init__.py
│   │   └── loaders.py          # "get_bunny()", "get_sphere()"
│   │
│   └── config.py               # Settings (e.g. USE_GPU = False)
│
├── tests/
├── examples/
├── pyproject.toml              # Modern Build System
└── CITATION.cff                # Academic Credit