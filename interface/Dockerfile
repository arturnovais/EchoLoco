FROM python:3.11-bookworm  

WORKDIR /app

COPY interface/requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY interface/. .          

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
