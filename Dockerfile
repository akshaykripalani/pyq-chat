# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install uv - a faster Python package installer
RUN pip install uv

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt using uv
RUN uv pip install --no-cache-dir -r requirements.txt --system

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (Optional: you might want to pass these at runtime)
# ENV GEMINI_API_KEY=
# ENV ANTHROPIC_API_KEY=
# ENV DISCORD_WEBHOOK_URL=

# Run uvicorn when the container launches
# Use 0.0.0.0 to allow connections from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
