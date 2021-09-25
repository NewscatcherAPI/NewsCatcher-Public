# Use Python36
FROM python:3.6

# Arguments
ARG directory
ARG number_workers

WORKDIR /app

# Copy requirements.txt to the docker image and install packages
COPY ./${directory}/requirements.txt ./
RUN pip install -r requirements.txt

# Copy all files in the folder app
COPY ./${directory}/ /app/

# Expose port 5000
EXPOSE 5000
ENV PORT 5000
ENV NB ${number_workers}

# Use gunicorn as the entrypoint
CMD exec gunicorn --bind :$PORT main:app --workers 1 --threads $NB --timeout 0