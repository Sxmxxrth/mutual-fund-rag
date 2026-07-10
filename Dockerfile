# Use the official Python 3.9 image for maximum ML compatibility
FROM python:3.9-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install system dependencies for C++ compilation (needed for ChromaDB/sentence-transformers)
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create necessary directories
RUN mkdir -p /code/data/chroma_db /code/temp_uploads

# Set up a non-root user (required by some cloud providers like HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Expose port 7860 for Gradio / HuggingFace Spaces
EXPOSE 7860

# Command to run the Gradio application
CMD ["python", "app.py"]
