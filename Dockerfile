FROM python:3.10-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ musl-dev curl libffi-dev gfortran libopenblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir poetry

# Copy the dependencies files
COPY pyproject.toml poetry.lock* ./

# Install the dependencies, including gunicorn and uvicorn
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-root \
    && pip install --no-cache-dir gunicorn uvicorn

# Create the final image
FROM python:3.10-slim

WORKDIR /app

# Copy the installed packages from the builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application and config
COPY r2r /app/r2r
COPY config.json /app/config.json

# Expose the port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

ENV PYTHONASYNCIODEBUG=1

# Run the application
#CMD ["uvicorn", "r2r.main.app_entry:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "r2r.main.app_entry:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
