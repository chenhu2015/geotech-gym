# Test image for the OpenFOAM task family (ESI OpenFOAM v2412 + Python, NumPy, SciPy, pytest).
#
#   podman build -t mechgym-foam-dev -f docker/foam-dev.Dockerfile docker
#   podman run --rm -v "$PWD":/work:z mechgym-foam-dev python3 -m pytest -p no:cacheprovider tests/test_foam.py
FROM docker.io/opencfd/openfoam-default:2412
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-numpy python3-scipy \
        python3-pytest && rm -rf /var/lib/apt/lists/*
ENTRYPOINT []
ENV PYTHONPATH=/work/src PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
