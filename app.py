import io
import base64
import math
import os
import re
import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import arxiv
import nltk
import yake
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data on startup (only once)
for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords

ENGLISH_STOPWORDS = set(stopwords.words("english"))
EXTRA_STOPWORDS = {
    # paper structure / meta
    "paper", "papers", "work", "works", "study", "studies", "article",
    "approach", "approaches", "method", "methods", "framework", "frameworks",
    "system", "systems", "technique", "techniques", "algorithm", "algorithms",
    "model", "models", "architecture", "architectures",
    # generic verbs used in abstracts
    "propose", "proposed", "proposes", "present", "presents", "presented",
    "introduce", "introduces", "introduced", "show", "shown", "shows",
    "demonstrate", "demonstrates", "demonstrated", "suggest", "suggests",
    "suggested", "indicate", "indicates", "indicated", "find", "found",
    "observe", "observed", "note", "noted", "highlight", "highlights",
    "highlighted", "achieve", "achieves", "achieved", "obtain", "obtains",
    "obtained", "improve", "improves", "improved", "outperform", "outperforms",
    "outperformed", "compare", "compared", "enable", "enables", "enabled",
    "explore", "explores", "explored", "investigate", "investigates",
    "investigated", "examine", "examines", "examined", "consider",
    "considers", "considered", "address", "addresses", "addressed",
    "extend", "extends", "extended", "develop", "develops", "developed",
    "design", "designs", "designed", "build", "builds", "built",
    "apply", "applied", "applies", "validate", "validates", "validated",
    "discuss", "discusses", "discussed", "require", "requires", "required",
    "provide", "provides", "provided", "describe", "describes", "described",
    # generic nouns
    "result", "results", "finding", "findings", "improvement", "improvements",
    "performance", "contribution", "contributions", "application", "applications",
    "analysis", "analyses", "evaluation", "evaluations", "comparison",
    "experiment", "experiments", "experimental", "baseline", "baselines",
    "dataset", "datasets", "benchmark", "benchmarks",
    "task", "tasks", "problem", "problems", "challenge", "challenges",
    "setting", "settings", "scenario", "scenarios", "case", "cases",
    "number", "numbers", "set", "sets", "type", "types", "kind", "kinds",
    # modifiers / connectives
    "using", "based", "also", "new", "use", "used", "two", "one", "first",
    "may", "however", "we", "our", "the", "this", "that", "these", "those",
    "can", "could", "would", "thus", "well", "high", "low", "large", "small",
    "data", "good", "better", "best", "much", "many", "more", "most",
    "less", "least", "even", "still", "without", "within", "across",
    "further", "due", "via", "per", "significantly", "existing", "recent",
    "state", "art", "train", "training", "test", "testing", "make", "makes",
    "made", "different", "various", "several", "show",
    # common idioms / filler that escape stopword lists
    "red tape", "cutting edge", "state art", "real world", "wide range",
    "large scale", "end end", "high quality", "low cost",
}
ALL_STOPWORDS = ENGLISH_STOPWORDS | EXTRA_STOPWORDS

ARXIV_CATEGORIES = {
    "Computer Science": [
        "cs.AI", "cs.CL", "cs.CV", "cs.GR", "cs.GT", "cs.HC", "cs.IR",
        "cs.IT", "cs.LG", "cs.MA", "cs.MM", "cs.NE", "cs.NI", "cs.PL",
        "cs.RO", "cs.SD", "cs.SE", "cs.SY",
    ],
    "Mathematics": [
        "math.AG", "math.AP", "math.CO", "math.GT", "math.NA",
        "math.OC", "math.PR", "math.ST",
    ],
    "Physics": [
        "physics.optics", "physics.comp-ph", "quant-ph", "cond-mat.str-el",
        "hep-th", "astro-ph.CO",
    ],
    "Statistics": ["stat.ML", "stat.ME", "stat.TH", "stat.CO"],
    "Electrical Engineering": ["eess.AS", "eess.IV", "eess.SP", "eess.SY"],
    "Quantitative Biology": ["q-bio.NC", "q-bio.QM"],
    "Economics": ["econ.EM", "econ.GN"],
}

app = FastAPI(title="arXiv Word Cloud")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response models ──────────────────────────────────────────────────

class WordCloudRequest(BaseModel):
    query: str
    start_date: Optional[str] = None   # "YYYY-MM-DD"
    end_date: Optional[str] = None     # "YYYY-MM-DD"
    categories: list[str] = []
    max_results: int = 200
    mode: str = "both"   # "words" | "phrases" | "both" | "trending"


class PaperInfo(BaseModel):
    title: str
    authors: list[str]
    published: str
    abstract: str
    url: str
    categories: list[str]


class WordCloudResponse(BaseModel):
    wordcloud_words: Optional[str] = None      # base64 PNG — n-gram frequencies
    wordcloud_phrases: Optional[str] = None    # base64 PNG — YAKE keyphrases
    wordcloud_trending: Optional[str] = None   # base64 PNG — trending enrichment
    papers: list[PaperInfo]
    total: int
    n_new: int = 0       # papers in "new" window (trending mode)
    n_old: int = 0       # papers in "baseline" window (trending mode)
    trending_window_new: str = ""
    trending_window_old: str = ""
    word_freq: dict[str, int]
    phrase_freq: dict[str, float]
    trending_terms: list[dict] = []  # top trending terms with scores for the table


# ── Text helpers ───────────────────────────────────────────────────────────────

def _query_stopwords(query: str) -> set[str]:
    """Suppress the search query's own words from all clouds."""
    words = re.sub(r"[^a-z\s]", " ", query.lower()).split()
    return {w for w in words if len(w) > 1}


# LaTeX command names that appear as bare tokens after stripping — add them
# to a blocklist so they never pollute word clouds.
_LATEX_TOKEN_BLOCKLIST = {
    "textsc", "textbf", "textit", "emph", "cite", "ref", "label",
    "begin", "end", "left", "right", "frac", "sqrt", "sum", "prod",
    "mathbb", "mathcal", "mathbf", "mathrm", "mathit", "mathsf",
    "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda",
    "sigma", "omega", "phi", "psi", "mu", "nu", "rho", "tau", "xi",
    "eta", "iota", "kappa", "zeta", "infty", "cdot", "times", "leq",
    "geq", "neq", "approx", "equiv", "subseteq", "subset", "cup", "cap",
    "forall", "exists", "nabla", "partial", "int", "oint", "hat", "bar",
    "tilde", "vec", "dot", "ddot", "overline", "underline", "item",
    "noindent", "centering", "hline", "multicolumn", "multirow",
    "footnote", "caption", "section", "subsection", "paragraph",
    "textcolor", "colorbox", "url", "href", "textwidth", "linewidth",
    "vspace", "hspace", "newline", "newpage", "clearpage",
    # common 2–3 letter LaTeX remnants and ambiguous acronyms
    "rtl", "ltr", "ie", "eg", "cf", "etc", "ibm", "mit", "pdf",
    "fig", "tab", "sec", "app", "equ", "def", "thm", "lem", "cor",
    "alg", "exp",
    # short tokens that are either names or unexpanded acronyms
    "tom",   # Theory of Mind acronym — the full phrase is more useful
    "sam",   # Segment Anything Model acronym
    "bob", "alice", "john",  # placeholder names in examples
    # single-char or numeric-only tokens are filtered by length check
}


def _clean_latex(text: str) -> str:
    """Strip LaTeX markup from arXiv abstracts before tokenization.

    arXiv abstracts frequently contain raw LaTeX such as $x_i$, \\textsc{Foo},
    \\cite{bar2023}, equation environments, etc. These produce garbage tokens
    like 'textsc', 'rtl', 'frac' in the word cloud.
    """
    # Remove display math: $$...$$ and \[...\]
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    # Remove inline math: $...$ and \(...\)
    text = re.sub(r"\$[^$]*?\$", " ", text)
    text = re.sub(r"\\\(.*?\\\)", " ", text)
    # Remove LaTeX commands with arguments: \cmd{...} or \cmd[...]{...}
    text = re.sub(r"\\[a-zA-Z]+\s*(\[[^\]]*\])?\s*\{[^}]*\}", " ", text)
    # Remove bare LaTeX commands: \cmd
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    # Remove remaining braces and special chars
    text = re.sub(r"[{}\[\]\\|^_~`<>]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize_flat(texts: list[str]) -> list[str]:
    """Clean LaTeX, lowercase, remove non-alpha, return flat token list."""
    tokens: list[str] = []
    for text in texts:
        text = _clean_latex(text)
        text = text.lower()
        text = re.sub(r"[^a-z\s\-]", " ", text)
        text = text.replace("-", " ")
        tokens.extend(text.split())
    return tokens


def _is_valid_token(t: str, all_sw: set[str]) -> bool:
    """True if a token is worth including in any word cloud."""
    return (
        len(t) > 2
        and t not in all_sw
        and t not in _LATEX_TOKEN_BLOCKLIST
        and not t.isdigit()
        and not re.fullmatch(r"[a-z]{1,2}", t)  # drop 1-2 char tokens missed by len check
    )


def _ngram_counter(tokens: list[str], all_sw: set[str]) -> tuple[Counter, Counter, Counter]:
    """Return (uni_freq, bi_freq, tri_freq) from a flat token list."""
    uni = Counter(t for t in tokens if _is_valid_token(t, all_sw))
    bi: list[str] = []
    tri: list[str] = []
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if _is_valid_token(a, all_sw) and _is_valid_token(b, all_sw):
            bi.append(f"{a} {b}")
        if i < len(tokens) - 2:
            c = tokens[i + 2]
            if _is_valid_token(a, all_sw) and _is_valid_token(b, all_sw) and _is_valid_token(c, all_sw):
                tri.append(f"{a} {b} {c}")
    return uni, Counter(bi), Counter(tri)


def _extract_ngram_frequencies(texts: list[str], extra_stopwords: set[str]) -> dict[str, int]:
    """N-gram frequencies with bigram/trigram boost."""
    all_sw = ALL_STOPWORDS | extra_stopwords
    tokens = _tokenize_flat(texts)
    uni, bi, tri = _ngram_counter(tokens, all_sw)

    result: dict[str, int] = dict(uni)
    for phrase, count in bi.items():
        if count >= 2 and phrase not in EXTRA_STOPWORDS:
            result[phrase] = result.get(phrase, 0) + count * 4
    for phrase, count in tri.items():
        if count >= 2 and phrase not in EXTRA_STOPWORDS:
            result[phrase] = result.get(phrase, 0) + count * 6
    return result


def _extract_keyphrases(texts: list[str], extra_stopwords: set[str]) -> dict[str, float]:
    """YAKE keyphrases — multi-word only, query words suppressed."""
    all_sw = ALL_STOPWORDS | extra_stopwords
    extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.6, top=30, features=None)
    phrase_scores: dict[str, float] = {}
    for text in texts:
        if not text.strip():
            continue
        for phrase, score in extractor.extract_keywords(_clean_latex(text)):
            phrase_lower = phrase.lower().strip()
            words = phrase_lower.split()
            if len(words) < 2:
                continue
            if any(w in all_sw for w in words):
                continue
            phrase_scores[phrase_lower] = phrase_scores.get(phrase_lower, 0) + 1.0 / (score + 1e-9)
    return phrase_scores


def _compute_trending_scores(
    new_texts: list[str],
    old_texts: list[str],
    extra_stopwords: set[str],
    min_new_count: int = 2,
) -> dict[str, float]:
    """Compute how much each term is enriched in new_texts vs old_texts.

    Score = enrichment_ratio × log(count_new + 1)

    enrichment_ratio = (p_new + ε) / (p_old + ε)
      where p = relative frequency within that corpus.

    Terms that are absent from old_texts but present in new_texts get a high
    ratio; terms that grew substantially also rank well. The log factor
    ensures noise terms (appearing once or twice) don't dominate.
    Multi-word phrases are boosted so concrete sub-topics beat single words.
    """
    all_sw = ALL_STOPWORDS | extra_stopwords
    eps = 1e-5  # smoothing prevents ÷0 and dampens extremely rare baseline terms

    new_tokens = _tokenize_flat(new_texts)
    old_tokens = _tokenize_flat(old_texts)

    new_uni, new_bi, new_tri = _ngram_counter(new_tokens, all_sw)
    old_uni, old_bi, old_tri = _ngram_counter(old_tokens, all_sw)

    n_new = max(sum(new_uni.values()), 1)
    n_old = max(sum(old_uni.values()), 1)

    scores: dict[str, float] = {}

    for new_dict, old_dict, phrase_boost in [
        (new_uni, old_uni, 1.0),
        (new_bi,  old_bi,  3.0),
        (new_tri, old_tri, 5.0),
    ]:
        for term, new_count in new_dict.items():
            if new_count < min_new_count:
                continue
            if term in EXTRA_STOPWORDS:
                continue
            old_count = old_dict.get(term, 0)
            p_new = new_count / n_new
            p_old = old_count / n_old
            enrichment = (p_new + eps) / (p_old + eps)
            # Only include terms that are actually more common recently
            if enrichment < 1.2:
                continue
            scores[term] = enrichment * math.log(new_count + 1) * phrase_boost

    return scores


# ── Word cloud renderer ────────────────────────────────────────────────────────

def _generate_wordcloud(frequencies: dict, colormap: str = "viridis") -> str:
    """Render a word cloud PNG and return as base64 string."""
    if not frequencies:
        return ""
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap=colormap,
        max_words=150,
        prefer_horizontal=0.8,
        min_font_size=10,
    ).generate_from_frequencies(frequencies)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Date helpers ───────────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fmt_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _trending_windows(start_date: Optional[str], end_date: Optional[str]) -> tuple[str, str, str, str]:
    """Return (new_start, new_end, old_start, old_end) for trending comparison.

    Strategy: baseline window = equally-long window immediately before new window.
    Minimum window = 30 days; maximum lookback is 6 months to keep results fresh.
    """
    today = datetime.utcnow()

    if end_date:
        new_end = _parse_date(end_date)
    else:
        new_end = today

    if start_date:
        new_start = _parse_date(start_date)
    else:
        new_start = new_end - timedelta(days=30)

    window = max(new_end - new_start, timedelta(days=7))
    # Cap baseline lookback so it doesn't go unreasonably far back
    window = min(window, timedelta(days=180))

    old_end = new_start
    old_start = old_end - window

    return _fmt_date(new_start), _fmt_date(new_end), _fmt_date(old_start), _fmt_date(old_end)


# ── arXiv fetch ────────────────────────────────────────────────────────────────

def _build_arxiv_query(
    query: str,
    categories: list[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> str:
    parts = []
    if query.strip():
        parts.append(f'(ti:"{query.strip()}" OR abs:"{query.strip()}")')
    if categories:
        parts.append("(" + " OR ".join(f"cat:{c}" for c in categories) + ")")
    if start_date or end_date:
        sd = start_date.replace("-", "") if start_date else "00000000"
        ed = end_date.replace("-", "") if end_date else "99991231"
        parts.append(f"submittedDate:[{sd}0000 TO {ed}2359]")
    return " AND ".join(parts) if parts else "all:*"


def _fetch_papers(
    query: str,
    categories: list[str],
    start_date: Optional[str],
    end_date: Optional[str],
    max_results: int,
) -> list[PaperInfo]:
    arxiv_query = _build_arxiv_query(query, categories, start_date, end_date)
    logger.info("arXiv query: %s (max=%d)", arxiv_query, max_results)
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=arxiv_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers: list[PaperInfo] = []
    for result in client.results(search):
        pub = result.published.strftime("%Y-%m-%d") if result.published else ""
        if start_date and pub and pub < start_date:
            continue
        if end_date and pub and pub > end_date:
            continue
        papers.append(PaperInfo(
            title=result.title,
            authors=[a.name for a in result.authors[:5]],
            published=pub,
            abstract=result.summary,
            url=result.entry_id,
            categories=result.categories,
        ))
    return papers


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path("static/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=index_path.read_text())


@app.get("/api/categories")
async def get_categories():
    return ARXIV_CATEGORIES


@app.post("/api/wordcloud", response_model=WordCloudResponse)
async def generate_wordcloud(req: WordCloudRequest):
    if not req.query.strip() and not req.categories:
        raise HTTPException(status_code=400, detail="Provide a search query or select at least one category.")

    max_results = max(10, min(req.max_results, 500))
    q_stop = _query_stopwords(req.query)

    # ── Trending mode ──────────────────────────────────────────────────────────
    if req.mode == "trending":
        new_start, new_end, old_start, old_end = _trending_windows(req.start_date, req.end_date)

        try:
            # Fetch both windows; halve max_results per window to stay within limits
            half = max(30, max_results // 2)
            new_papers = _fetch_papers(req.query, req.categories, new_start, new_end, half)
            old_papers = _fetch_papers(req.query, req.categories, old_start, old_end, half)
        except Exception as e:
            logger.error("arXiv fetch error: %s", e)
            raise HTTPException(status_code=502, detail=f"arXiv API error: {str(e)}")

        if not new_papers and not old_papers:
            raise HTTPException(status_code=404, detail="No papers found for the given criteria.")

        new_texts = [f"{p.title} {p.abstract}" for p in new_papers]
        old_texts = [f"{p.title} {p.abstract}" for p in old_papers]

        trending_scores = _compute_trending_scores(new_texts, old_texts, q_stop)
        top_trending = dict(sorted(trending_scores.items(), key=lambda x: -x[1])[:200])
        wc_trending = _generate_wordcloud(top_trending, colormap="YlOrRd")

        # Build ranked table for UI
        trending_table = [
            {"term": t, "score": round(s, 2)}
            for t, s in sorted(trending_scores.items(), key=lambda x: -x[1])[:40]
        ]

        all_papers = new_papers + old_papers
        return WordCloudResponse(
            wordcloud_trending=wc_trending,
            papers=all_papers,
            total=len(all_papers),
            n_new=len(new_papers),
            n_old=len(old_papers),
            trending_window_new=f"{new_start} → {new_end}",
            trending_window_old=f"{old_start} → {old_end}",
            word_freq={},
            phrase_freq={},
            trending_terms=trending_table,
        )

    # ── Standard modes (words / phrases / both) ────────────────────────────────
    try:
        papers = _fetch_papers(req.query, req.categories, req.start_date, req.end_date, max_results)
    except Exception as e:
        logger.error("arXiv fetch error: %s", e)
        raise HTTPException(status_code=502, detail=f"arXiv API error: {str(e)}")

    if not papers:
        raise HTTPException(status_code=404, detail="No papers found for the given criteria.")

    all_texts = [f"{p.title} {p.abstract}" for p in papers]
    all_abstracts = [p.abstract for p in papers]

    wordcloud_words_b64: Optional[str] = None
    wordcloud_phrases_b64: Optional[str] = None
    ngram_freq: dict[str, int] = {}
    phrase_freq: dict[str, float] = {}

    if req.mode in ("words", "both"):
        ngram_freq = _extract_ngram_frequencies(all_texts, q_stop)
        top_ngrams = dict(sorted(ngram_freq.items(), key=lambda x: -x[1])[:300])
        wordcloud_words_b64 = _generate_wordcloud(top_ngrams, colormap="Blues")

    if req.mode in ("phrases", "both"):
        phrase_freq = _extract_keyphrases(all_abstracts, q_stop)
        wordcloud_phrases_b64 = _generate_wordcloud(phrase_freq, colormap="Oranges")

    return WordCloudResponse(
        wordcloud_words=wordcloud_words_b64,
        wordcloud_phrases=wordcloud_phrases_b64,
        papers=papers,
        total=len(papers),
        word_freq={k: v for k, v in sorted(ngram_freq.items(), key=lambda x: -x[1])[:50]},
        phrase_freq={k: round(v, 2) for k, v in sorted(phrase_freq.items(), key=lambda x: -x[1])[:200]},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("ENV", "development") != "production"
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload)
