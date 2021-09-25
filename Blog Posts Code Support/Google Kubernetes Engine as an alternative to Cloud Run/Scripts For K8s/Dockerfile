# Use Python36
FROM python:3.6

# Arguments to use in docker run
ARG directory

# Copy requirements.txt to the docker image and install packages
COPY ./${directory}/. /app
WORKDIR /app
RUN pip install -r requirements.txt

# Use gunicorn as the entrypoint
CMD [ "python", "./main.py" ]