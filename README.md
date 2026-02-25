# DTS AI Engine

AI-powered Document Tracking Assistant — a conversational REST API that helps users check document status using natural language.

## Features

- 🤖 **ML-Based Intent Classification** — TF-IDF + SGDClassifier that you can retrain anytime
- 🔍 **Entity Extraction** — Automatically detects PDID numbers in various formats
- 💬 **Multi-Turn Conversations** — Remembers context across messages (e.g., asks for PDID, then uses it)
- 📊 **Retrainable** — Add new training data via CSV or database, then retrain via API
- 🗄️ **Chat Logging** — All conversations stored in MySQL for analysis
- 🔌 **DTS Integration Ready** — Mock mode for development, switch to real DTS API via config

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env with your MySQL credentials
```

### 3. Create Database

```sql
mysql -u root -p < schema.sql
```

Or the engine will auto-create tables on first startup.

### 4. Train the Model

```bash
python scripts/train.py
```

Or skip this — the engine auto-trains on startup if no model exists.

### 5. Run the Server

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the Swagger UI.

## API Endpoints

| Method | Endpoint     | Description                     |
| ------ | ------------ | ------------------------------- |
| POST   | `/ai/chat`   | Send a message, get AI response |
| POST   | `/ai/train`  | Retrain the ML model            |
| GET    | `/ai/health` | Health check + model status     |

### Chat Example

```bash
# Ask about document status
curl -X POST http://localhost:8000/ai/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the status of my document?"}'

# Response: {"reply": "What is the PDID of your document?", "session_id": "abc-123", ...}

# Provide PDID (use session_id from previous response)
curl -X POST http://localhost:8000/ai/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "PDID 001", "session_id": "abc-123"}'

# Response: {"reply": "📄 Document Status for PDID 001\n• Status: Processing\n• Current Department: HR\n...", ...}
```

### Retrain the Model

```bash
curl -X POST http://localhost:8000/ai/train \
  -H "Content-Type: application/json" \
  -d '{"source": "csv"}'
```

## Adding Training Data

### Option 1: Edit the CSV

Add new rows to `ml_data/intent_training.csv`:

```csv
text,intent
"Where is my leave form?",document_status
"PDID 100",follow_up
```

Then retrain: `POST /ai/train {"source": "csv"}`

### Option 2: Insert into Database

```sql
INSERT INTO training_data (text, intent) VALUES
('Where is my leave form?', 'document_status'),
('Check PDID 200 please', 'document_status');
```

Then retrain: `POST /ai/train {"source": "database"}`

## Switching to Real DTS API

When the DTS team provides their API, update `.env`:

```env
DTS_MOCK_MODE=false
DTS_API_BASE_URL=http://your-dts-server:8080
```

The engine expects `GET {DTS_API_BASE_URL}/api/documents/{pdid}` to return document info as JSON.

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
DTS_AI/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Environment settings
│   ├── api/routes.py        # REST endpoints
│   ├── ml/                  # ML pipeline
│   │   ├── intent_classifier.py
│   │   ├── entity_extractor.py
│   │   └── preprocessing.py
│   ├── services/            # Business logic
│   │   ├── conversation.py  # Chat orchestrator
│   │   ├── dts_client.py    # DTS API client
│   │   ├── response_generator.py
│   │   └── chat_logger.py
│   ├── db/                  # Database
│   │   ├── models.py
│   │   └── database.py
│   └── schemas/chat.py      # Pydantic models
├── ml_data/                 # Training data
├── ml_models/               # Saved models
├── scripts/train.py         # Training CLI
├── tests/                   # Test suite
└── schema.sql               # MySQL DDL
```
