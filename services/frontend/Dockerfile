# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy just the requirements file first
COPY ./requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy your application code into the container
COPY ./app.py /app/app.py

# Command to run the Streamlit app when the container starts
# We expose it on port 8501 and make it accessible
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]