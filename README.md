# arXiv Word Cloud

A web app that searches arXiv papers by topic, date range, and category, then visualizes research trends as a word cloud — using both raw word frequencies and extracted keyphrases.

## Features

- **Topic search** — searches arXiv titles and abstracts
- **Date range filter** — narrow to specific time windows
- **Category filter** — filter by arXiv subject areas (cs.LG, cs.CV, math.ST, etc.)
- **Two word cloud modes**:
  - **Raw Words** — tokenized word frequencies (blue palette)
  - **Keyphrases** — key multi-word phrases extracted with YAKE (orange palette)
  - **Both** — show both side by side
- **Paper list** — collapsible list of matched papers with titles, authors, dates, categories, and abstract previews

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.10+ recommended.

### 2. Run the app

```bash
python app.py
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

Alternatively, with auto-reload for development:

```bash
uvicorn app:app --reload --port 8000
```

## Usage

1. Enter a **search topic** (e.g. `diffusion models`, `graph neural networks`, `reinforcement learning`)
2. Set a **date range** (defaults to the last 6 months)
3. Optionally select one or more **arXiv categories** (e.g. `cs.LG`, `cs.CV`)
4. Adjust **Max Papers** (10–500; more papers = more representative cloud but slower)
5. Choose a **word cloud mode**: Both / Raw Words / Keyphrases
6. Click **Generate Word Cloud**

## Project Structure

```
Arxiv-word-cloud/
├── app.py              # FastAPI backend
├── requirements.txt    # Python dependencies
├── README.md
└── static/
    ├── index.html      # Single-page UI
    ├── style.css       # Styling
    └── app.js          # Frontend logic
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the web UI |
| `GET` | `/api/categories` | List of arXiv categories |
| `POST` | `/api/wordcloud` | Search papers and generate word clouds |

### POST `/api/wordcloud` payload

```json
{
  "query": "transformer attention",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "categories": ["cs.LG", "cs.CV"],
  "max_results": 200,
  "mode": "both"
}
```

`mode` can be `"words"`, `"phrases"`, or `"both"`.

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Web server |
| `arxiv` | arXiv API client |
| `wordcloud` | Word cloud image generation |
| `nltk` | Stopword removal and tokenization |
| `yake` | Keyphrase extraction (no model download required) |
| `Pillow` | Image processing |

## Notes

- arXiv API has rate limits; larger result sets (300–500) may take 30–60 seconds.
- NLTK stopwords are downloaded automatically on first run.
- The YAKE keyphrase extractor runs entirely offline, no API key needed.
