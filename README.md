# EntreLocate
EntreLocate is an AI-powered business location advisory tool designed to assist entrepreneurs and investors in evaluating the potential of opening a business in specific districts across Türkiye. The system leverages retrieval-augmented generation (RAG), real socioeconomic data, and semantic search to provide risk scores and smart location recommendations.

# Features:
\n Smart Location Recommendation: Suggests the most suitable districts for new businesses based on population demographics, age and gender distribution, and existing business statistics.
\n Risk Score Generation: Calculates a risk score (0–100) by evaluating competition density, socioeconomic levels, and regional growth trends.
\n Interactive Chatbot: Uses large language models (LLMs) to answer user queries in both Turkish and English, and provides insights in natural language.
\n Map Interface: Visualizes recommended locations and nearby facilities using OpenStreetMap and the Overpass API.
\n Document Processing: Extracts relevant statistical data from PDF and Excel files to support grounded and accurate AI-generated responses.

#Tech Stack
\n Frontend: Gradio, HTML, Leaflet.js
\n Backend: Python, Flask
\n AI/ML: Google Gemini API, SentenceTransformers (MiniLM-L6-v2)
\n Database: ChromaDB for vector storage and semantic retrieval
\n Other Tools: dotenv, PyMuPDF, OpenStreetMap, Overpass API



