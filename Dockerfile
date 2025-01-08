FROM python:3.10


COPY . /app

WORKDIR /app

RUN chmod +x vuljavadetectmodel/getjoern.sh 
RUN chmod +x vuljavadetectmodel/getreference.sh

# download joern and reference files (model checkpoint)
RUN vuljavadetectmodel/getjoern.sh
RUN vuljavadetectmodel/getreference.sh


RUN pip install --upgrade pip && \
    pip install -r requirements.txt


ENV PORT 8000

EXPOSE $PORT
	
# Start 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$PORT", "--workers", "4"]
	