# WorkPulse — Local Setup Guide

Complete guide to run WorkPulse locally on your machine.

---

## Prerequisites

- Python 3.9+
- Node.js 18+
- Git
- A HuggingFace account (free)

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/imperfectly-perfect1003/internship-2026.git
cd internship-2026
```

---

## Step 2 — Run the Backend

```bash
cd Week5-NLP-WorkPulse

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn transformers torch python-dotenv
```

Create a `.env` file inside `Week5-NLP-WorkPulse/`:
HF_MODEL=Madhuri1003/workpulse-distilbert
Start the backend:

```bash
uvicorn main:app --reload
```

Backend runs at http://127.0.0.1:8000
API docs at http://127.0.0.1:8000/docs

---

## Step 3 — Run the Frontend

Open a new terminal:

```bash
cd internship-2026/Week7-FullStack-WorkPulse

# Install dependencies
npm install

# Start development server
npm start
```

Frontend runs at http://localhost:3000

---

## Step 4 — Connect Frontend to Local Backend

In `Week7-FullStack-WorkPulse/src/App.js`, change line 4:

```javascript
// Change this (production):
const API_URL = "https://madhuri1003-workpulse-api.hf.space";

// To this (local):
const API_URL = "http://127.0.0.1:8000";
```

---

## Live Deployment

If you just want to use the live app without running locally:

- Frontend: https://workpulse-frontend-pied.vercel.app
- API: https://madhuri1003-workpulse-api.hf.space/docs
- Model: https://huggingface.co/Madhuri1003/workpulse-distilbert