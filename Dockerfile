# Use a base image with Python and necessary dependencies
FROM python:3.8-slim-buster

# Install system library required for Tkinter
RUN apt-get update && \
    apt-get install -y tk

# Set the working directory
WORKDIR /app

# Introduce a label for cache busting the context history
LABEL BUST_CONTEXT=${BUILD_DATE}

# Copy the requirements file into the container
COPY requirements.txt .
COPY conversation_log.txt .
COPY bot.py .

# Install necessary Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install Levenshtein
RUN pip install requests
#RUN pip install matplotlib
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install nltk
RUN pip install termcolor
#RUN pip install pygame
RUN pip install wikipedia

# Install ROCm dependencies if ROCM environment variable is set
# ARG ROCM
# RUN if [ "$ROCM" = "true" ]; then \
#     apt-get update && \
#     apt-get install -y rocm-libs miopen-hip; \
#     pip install numba; \
# fi

# Install CUDA dependencies if CUDA environment variable is set
# ARG CUDA
# RUN if [ "$CUDA" = "true" ]; then \
#     apt-get update && \
#     apt-get install -y nvidia-cuda-toolkit; \
#     pip install numba; \
# fi

# Install NLTK and download required datasets
RUN python -m nltk.downloader words brown

# Set NLTK_DATA to the current directory
ENV NLTK_DATA=/

# Run the bot's script when the container starts
CMD ["python", "bot.py", "simulate_conversation"]
