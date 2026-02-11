FROM python:3.10-slim

# Install system dependencies for GIS and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    g++ \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL if building from source is needed
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set the PYTHONPATH to include src for easy imports
ENV PYTHONPATH=/app/src

# Default entrypoint for the pipeline
ENTRYPOINT ["python", "src/run_pipeline.py"]
