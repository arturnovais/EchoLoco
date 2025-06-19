# Vector Database - EchoLoco

Este módulo contém os serviços para gerenciamento e popularização do banco de dados vetorial com embeddings de áudios utilizando **Qdrant** e **Titanet (NVIDIA)**.

---

## 📂 Estrutura relevante

```
src/services/vector_database/
├── qdrant_service.py        # Serviço que gerencia o Qdrant
└── populate_database.py     # Script para popular o banco com embeddings dos áudios
```

---

## 🐳 Inicializar o banco de dados vetorial (Qdrant)

1️⃣ Na raiz do projeto (`EchoLoco/`), rode:
```
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
Isso irá subir um container do Qdrant em `localhost:6334`.

---

## 📥 Popular a base de dados com os áudios vetorizados

Certifique-se de que:
- O diretório `audios/` contém os áudios conforme especificado no `audios_metadata.csv`
- O `audios_metadata.csv` possui caminhos relativos ao diretório `audios/` na coluna `path` (exemplo: `Artur/7_sr.wav`)

2️⃣ Execute o script para popular:
```
python -m src.services.vector_database.populate_database
```
O script irá:
- Carregar o modelo Titanet
- Criar (ou recriar) a collection no Qdrant
- Gerar embeddings dos áudios
- Inserir os embeddings no Qdrant com os metadados (path, ruído, transcrição)

## 🔍 Verificando as collections (opcional)

Você pode verificar as collections no Qdrant acessando a API REST:
```
http://localhost:6334/collections
```
ou usando o método:
```
qdrant_service.list_collections()
```

---

## 📌 Exemplo de execução esperada

```
Modelo carregado.
Processando: EchoLoco/audios/Artur/7_sr.wav
Embedding inserido com ID: f2c9a9e6-...
Processando: EchoLoco/audios/Artur/8_sr.wav
Embedding inserido com ID: 1a7f2d4b-...
```