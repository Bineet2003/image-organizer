FROM tensorflow/tensorflow:2.9.1
WORKDIR /app
COPY src/ /app/
RUN pip install -r /app/requirements.txt
CMD ["python", "/app/organize_images.py", "/mounted_data"]
