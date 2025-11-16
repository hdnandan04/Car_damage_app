# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Make port 7860 available to the world outside this container
# Hugging Face Spaces expects web apps to run on this port
EXPOSE 7860

# Define the command to run your app using Gunicorn
# This will run on port 7860, which is required by Hugging Face
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]