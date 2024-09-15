# Use the official Python image from Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV (including libGL)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install git and any other required system packages
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file from the current directory (host) to /app in the container
COPY requirements.txt .

# Install the required packages in the container
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application files to /app
COPY . /app

# Start the container with an interactive shell
CMD ["/bin/bash"]
