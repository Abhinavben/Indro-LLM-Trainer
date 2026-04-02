# Dockerfile for Indro-LLM-Trainer with CUDA Support

FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies (customize as needed)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && pip3 install -r requirements.txt

# Expose any ports the app requires (if applicable)
EXPOSE 8000

# Run the application
CMD [ "python3", "your_script.py" ]  # Replace with your actual script
