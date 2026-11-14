# Test image for the Code_Aster task family (official Code_Aster image + pytest + a writable /app).
#
#   podman build -t mechgym-aster-dev -f docker/aster-dev.Dockerfile docker
#   podman run --rm -v "$PWD":/work:z mechgym-aster-dev bash -c \
#       '. /opt/activate.sh; export PYTHONPATH=/work/src:/opt/pylib:$PYTHONPATH; python3 -m pytest -p no:cacheprovider tests/test_aster.py'
#
# (/opt/activate.sh sets PYTHONPATH for Code_Aster; extend it after sourcing.)
FROM docker.io/simvia/code_aster:stable
USER root
RUN mkdir -p /app /logs/verifier && chmod 777 /app /logs /logs/verifier \
    && bash -c '. /opt/activate.sh >/dev/null 2>&1 && python3 -m pip install --no-cache-dir --target /opt/pylib pytest==8.4.1'
USER user
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
