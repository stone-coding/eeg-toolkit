FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg \
    MNE_DONTWRITE_HOME=true MNE_DATA=/app/data

WORKDIR /app
# build deps are not needed: all scientific dependencies ship manylinux wheels
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[api,gui,s3]"

# paths in configs/default.yaml are relative to /app
VOLUME ["/app/data", "/app/outputs"]
EXPOSE 8000 8501

# default: the API; docker-compose overrides the command for the GUI
CMD ["eegkit", "serve", "-c", "configs/default.yaml", "--host", "0.0.0.0", "--port", "8000"]
