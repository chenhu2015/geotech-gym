# Development / test image: the verifier library plus OpenSeesPy, used to run the full test suite.
#
#   podman build -t mechgym-dev -f docker/mechgym.Dockerfile .
#   podman run --rm -v "$PWD":/work:z mechgym-dev pytest            # or: docker run ...
#
# Simulation software is only ever installed in containers, never on the host.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 matplotlib==3.10.3 openseespy==3.8.0.0 \
        pytest==8.4.1
ENV PYTHONPATH=/work/src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
WORKDIR /work
