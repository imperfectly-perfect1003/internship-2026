# WorkPulse — React Frontend

> Full-stack React application for company culture sentiment analysis, connected to a fine-tuned DistilBERT model via FastAPI.

## 🔗 Live Demo
**https://workpulse-frontend-pied.vercel.app**

---

## Features
- Single review analysis with confidence breakdown bars
- Batch analysis for up to 20 reviews at once
- Mixed sentiment detection with split-part visualization
- Real-time results with loading states and error handling
- Responsive dark UI

---

## Tech Stack
- React 18 (Create React App)
- Axios for API calls
- Deployed on Vercel

---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/imperfectly-perfect1003/internship-2026.git
cd internship-2026/Week7-FullStack-WorkPulse

# Install dependencies
npm install

# Start development server
npm start
```

The app runs on `http://localhost:3000` and connects to the live HuggingFace Spaces backend by default.

---

## Project Structure
Week7-FullStack-WorkPulse/
├── src/
│   ├── App.js        # Main component — all UI and API logic
│   └── App.css       # Styling
├── public/
│   └── index.html
└── package.json

---

## API Connection
The frontend connects to:
https://madhuri1003-workpulse-api.hf.space

Endpoints used:

- `POST /predict` — single review analysis
- `POST /batch` — batch analysis

---

## Backend
The FastAPI backend code is in `Week5-NLP-WorkPulse/`. See that README for full API documentation and model details.