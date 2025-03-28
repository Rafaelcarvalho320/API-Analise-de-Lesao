# Imagem base com Python 3.11
FROM python:3.11-slim 

# Diretório de trabalho dentro do container
WORKDIR /app

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copiar o requirements.txt para o container
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar os arquivos de modelo para o container
COPY best_xgb_model_81.joblib /app/best_xgb_model_81.joblib
COPY resnet18_improved78.h5 /app/resnet18_improved78.h5

# Copiar o restante do projeto para o container
COPY . .

# Expor a porta 9000 
EXPOSE 9000

# Comando para iniciar o servidor Django
CMD ["python", "myproject/manage.py", "runserver", "0.0.0.0:9000"]