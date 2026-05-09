"""generate_script.py

Module for generating narrative script segments using semantic document retrieval and LLM.

Description:
    Creates detailed narration script for video by combining semantic retrieval with
    iterative LLM generation. Processes project source materials (PDFs, text files,
    Wikipedia/web references) into semantic chunks, builds a FAISS vector index,
    retrieves relevant context per section, then uses DeepSeek to generate coherent
    narration while maintaining story continuity across sections.

Inputs:
    - outputs/output_jsons/outline_texts.json: Section outlines and titles
    - source_material/config.json: Project metadata (title, narration style, links)
    - source_material/*: Local PDFs and text files
    - Reference URLs from config.json

Outputs:
    - outputs/output_jsons/narration.json: Structured narration with sections and segmented narration_text
    - source_material/{project}_chunks.json: Cached text chunks (created once)
    - source_material/{project}_faiss.index: Vector index (created once)

Environment Variables:
    - DEEPSEEK_API_KEY: API key for DeepSeek language model

Configuration:
    - MIN_CHUNK_SIZE: 1000 characters per semantic chunk
    - FAISS model: 'BAAI/bge-large-en-v1.5' (semantic embeddings)
    - LLM model: 'deepseek-chat' / 'deepseek-reasoner' (for narration generation)
    - Temperature: 1.2 (creative generation)

Usage:
    python generate_script.py <project_name>
    
    Arguments:
        project_name: Project identifier (e.g., 'VikramBetaal')
"""

import os
import sys
import json
import re
import gc
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
import wikipediaapi
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import time

# ========================================================
# Helper: Get project name and paths from config
# ========================================================
def load_project_config(project_arg=None):
    """Load project name from config.json or command-line arg."""
    project = project_arg or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not project:
        raise ValueError("Project name required: python script.py PROJECT_NAME")
    
    # Scripts are in PROJECT_NAME/scripts/, so go up one level to PROJECT_NAME/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.join(base_dir, project)
    source_dir = os.path.join(project_root, "source_material")
    config_path = os.path.join(source_dir, 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    project_name = config.get("_project_config", {}).get("project_name", project)
    return project_name, source_dir, config

# ========================================================
# ------------------ CONFIGURATION -----------------------
# ========================================================
MODEL_NAME = "deepseek-reasoner" # LLM for narration generation: "deepseek-chat" "deepseek-reasoner
RETRIEVAL_MODEL = "BAAI/bge-small-en-v1.5"
MIN_CHUNK_SIZE = 1000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=ENV_PATH)
my_tkn = os.getenv("DEEPSEEK_API_KEY")
if not my_tkn:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env file")

project, source_dir, config = load_project_config()
project_root = os.path.dirname(source_dir)
json_dir = os.path.join(project_root, "outputs", "output_jsons")
os.makedirs(json_dir, exist_ok=True)


faiss_path = os.path.join(source_dir, f"{project}_faiss.index")
chunks_path = os.path.join(source_dir, f"{project}_chunks.json")
chunks_meta_path = os.path.join(source_dir, f"{project}_chunks.meta.json")
outline_json_path = os.path.join(json_dir, 'outline_texts.json')


# ========================================================
# ---------------- UTILITY FUNCTIONS ---------------------
# ========================================================

def get_wikipedia_text(title):
    """
    Fetch and return text content from Wikipedia page.
    
    Args:
        title: Wikipedia page title
    
    Returns:
        str: Full page text if found, None otherwise
    """    
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="NarrationGenerator/1.0")
    page = wiki.page(title)
    return page.text if page.exists() else None


def get_webpage_text(url):
    """
    Fetch and extract text from a web page.
    
    Args:
        url: URL to fetch
    
    Returns:
        str: Extracted text content, empty string if fetch fails
    
    Notes:
        - Uses BeautifulSoup for HTML parsing
        - 10-second timeout per request
    """    
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""


def read_txt(file_path):
    """
    Read plain text file content.
    
    Args:
        file_path: Path to text file
    
    Returns:
        str: File content
    """    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(file_path):
    """
    Extract text from PDF file.
    
    Args:
        file_path: Path to PDF file
    
    Returns:
        str: Combined text from all pages
    """    
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])


def split_into_chunks(chunks, text, min_chunk_size=MIN_CHUNK_SIZE):
    """Split text into semantic chunks for embedding and retrieval.
    
    Args:
        chunks: List to append chunks to
        text: Full text to split
        min_chunk_size: Minimum characters per chunk (default: 1000)
    
    Returns:
        list: Updated chunks list with new chunks appended
    
    Notes:
        - Splits on paragraph boundaries to preserve semantic coherence
        - Ensures each chunk meets minimum size requirement
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    temp_chunk = ""
    for para in paragraphs:
        if not temp_chunk:
            temp_chunk = para
        elif len(temp_chunk) + len(para) < min_chunk_size:
            temp_chunk += " " + para
        else:
            chunks.append(temp_chunk.strip())
            temp_chunk = para
    if temp_chunk:
        if len(temp_chunk) < min_chunk_size and chunks:
            last = chunks.pop()
            chunks.append((last + " " + temp_chunk).strip())
        else:
            chunks.append(temp_chunk.strip())
    return chunks


def process_all_documents(source_dir, reference_links):
    """Process all source documents and external references into text chunks.
    
    Args:
        source_dir: Directory containing local PDFs and text files
        reference_links: List of URLs (Wikipedia or web pages)
    
    Returns:
        list: Combined list of text chunks from all sources
    
    Notes:
        - Processes .txt and .pdf files from source_dir
        - Fetches and processes Wikipedia and web pages
        - Each chunk respects MIN_CHUNK_SIZE requirement
    """
    all_chunks = []
    for file in os.listdir(source_dir):
        path = os.path.join(source_dir, file)
        if file.endswith(".txt") and file not in ["all_links.txt", "outline_texts.txt"]:
            text = read_txt(path)
        elif file.endswith(".pdf"):
            text = read_pdf(path)
        else:
            continue

        chunks = []
        doc_chunks = split_into_chunks(chunks, text)
        all_chunks.extend(doc_chunks)

    for url in reference_links:
        if "wikipedia.org/wiki/" in url:
            title = url.split("/wiki/")[-1].replace("_", " ")
            fetched = get_wikipedia_text(title)
        else:
            fetched = get_webpage_text(url)

        if fetched:
            print("Fetched:", url)
            chunks = []
            doc_chunks = split_into_chunks(chunks, fetched)
            all_chunks.extend(doc_chunks)

    return all_chunks


def build_chunk_cache_metadata(source_dir, reference_links):
    """Capture enough local state to know when cached chunks are stale."""
    tracked_files = []
    for file in sorted(os.listdir(source_dir)):
        path = os.path.join(source_dir, file)
        if not os.path.isfile(path):
            continue
        if file == "config.json" or file.endswith(".txt") or file.endswith(".pdf"):
            tracked_files.append({
                "name": file,
                "mtime": os.path.getmtime(path),
                "size": os.path.getsize(path),
            })

    return {
        "reference_links": reference_links,
        "tracked_files": tracked_files,
        "min_chunk_size": MIN_CHUNK_SIZE,
    }


def chunk_cache_is_stale(source_dir, reference_links, chunks_path, chunks_meta_path):
    """Return True when chunks cache is missing or local inputs changed."""
    if not os.path.exists(chunks_path) or not os.path.exists(chunks_meta_path):
        return True

    try:
        with open(chunks_meta_path, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
    except Exception:
        return True

    current_meta = build_chunk_cache_metadata(source_dir, reference_links)
    return cached_meta != current_meta


def tokenize(text):
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower()) if len(t) > 2]


def lexical_score(query, text):
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    c_tokens = set(tokenize(text))
    return len(q_tokens & c_tokens) / len(q_tokens)


def hybrid_retrieve(query, faiss_index, model, chunks, k=6, dense_pool=24, lexical_pool=18):
    """Dense retrieval + lightweight lexical rerank + diversity filter."""
    if not chunks:
        return ""

    query_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    dense_scores, dense_ids = faiss_index.search(query_emb, min(dense_pool, len(chunks)))

    dense_map = {}
    for score, idx in zip(dense_scores[0], dense_ids[0]):
        if 0 <= idx < len(chunks):
            # cosine similarity is in [-1, 1], normalize to [0, 1]
            dense_map[idx] = (float(score) + 1.0) / 2.0

    lexical_scored = sorted(
        ((lexical_score(query, chunk), idx) for idx, chunk in enumerate(chunks)),
        reverse=True
    )[:lexical_pool]

    candidate_ids = set(dense_map.keys()) | {idx for _, idx in lexical_scored}
    ranked = []
    for idx in candidate_ids:
        lex = lexical_score(query, chunks[idx])
        dense = dense_map.get(idx, 0.0)
        ranked.append((0.75 * dense + 0.25 * lex, idx))
    ranked.sort(reverse=True)

    selected = []
    selected_token_sets = []
    for _, idx in ranked:
        chunk = chunks[idx]
        c_tokens = set(tokenize(chunk))
        too_similar = False
        for prev in selected_token_sets:
            overlap = len(c_tokens & prev) / max(1, len(c_tokens | prev))
            if overlap > 0.85:
                too_similar = True
                break
        if too_similar:
            continue
        selected.append(chunk)
        selected_token_sets.append(c_tokens)
        if len(selected) >= k:
            break

    return "\n\n".join(selected)


def update_story_summary(client, story_summary, new_response):
    """Update cumulative story summary with newly generated section.
    
    Args:
        client: OpenAI client for DeepSeek API
        story_summary: Previous cumulative summary (2-4 sentences)
        new_response: Newly generated section content
    
    Returns:
        str: Updated summary (2-4 sentences) maintaining narrative coherence
    
    Purpose:
        Passed to subsequent section generation to maintain continuity
    """
    
    prompt = f"""
You are maintaining a concise running summary of a YouTube narration script.
Here is the summary so far:
{story_summary}

Here is the newly generated section:
{new_response}

Update the summary to reflect everything so far, in 3 to 6 sentences.
Keep it brief but concise, coherent, dense, information-rich; preserving continuity and key narrative points.
"""
    summary_resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return summary_resp.choices[0].message.content.strip()


def make_outline_with_status(outline_data, current_index):
    """Create outline with completion status flags for LLM context.
    
    Args:
        outline_data: List of section outline dictionaries
        current_index: Index of currently generating section
    
    Returns:
        str: JSON string with 'done' and 'pending' status flags
    
    Notes:
        Helps LLM understand which sections are complete vs pending
    """
    updated = []
    for j, s in enumerate(outline_data):
        entry = s.copy()
        entry["status"] = "done" if j < current_index else "pending"
        updated.append(entry)
    return json.dumps(updated, ensure_ascii=False, indent=2)


def chat_with_llm(system_prompt, queries, outline_data, my_tkn):
    """Generate narration sections iteratively with story continuity.
    
    Args:
        system_prompt: System prompt defining generation style and format
        queries: List of per-section queries with context
        outline_data: Full outline with section titles and outlines
        my_tkn: DeepSeek API key
    
    Returns:
        list: JSON-formatted narration responses (one per section)
    
    Process:
        - For each section: retrieves relevant context, generates narration
        - Maintains running story_summary to ensure continuity
        - Passes section status to LLM for context awareness
    """
    client = OpenAI(api_key=my_tkn, base_url="https://api.deepseek.com")
    story_summary = ""
    responses = []

    for i, query in enumerate(queries):
        outline_with_status = make_outline_with_status(outline_data, i)
        contextual_query = f"""
{query}

EARLIER_SECTIONS_SUMMARY:
{story_summary}

FULL_OUTLINE_WITH_STATUS:
{outline_with_status}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": contextual_query},
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=1.2,
            messages=messages,
        )

        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```json|```$", "", content.strip(), flags=re.MULTILINE)
        responses.append(content)
        story_summary = update_story_summary(client, story_summary, content)
        print(f"[Section {i+1}/{len(queries)} complete]\n")

    return responses


# ========================================================
# ---------------- MAIN EXECUTION ------------------------
# ========================================================

# Load outline data
if not os.path.exists(outline_json_path):
    raise FileNotFoundError(f"Missing {outline_json_path}")


with open(outline_json_path, "r", encoding="utf-8") as f:
    outline_data = json.load(f)

sections = [s["section_title"] for s in outline_data]
outlines = [s["outline"] for s in outline_data]
n_section = len(outline_data)
print(f"Loaded {n_section} sections from outline_texts.json\n")

# Load description metadata
video_title = config["video_title"]
narration_style = "\n".join(config["narration_style"])
n_section = config["n_section"]
reference_links = config["reference_links"]
print(f"title: {video_title}\n narration_style: {narration_style}\nSections expected: {n_section}\n")

# Prepare chunks
if not chunk_cache_is_stale(source_dir, reference_links, chunks_path, chunks_meta_path):
    print(f"[LOAD] Using cached chunks: {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
else:
    print("[BUILD] Processing documents and creating chunks...")
    chunks = process_all_documents(source_dir, reference_links)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(chunks_meta_path, "w", encoding="utf-8") as f:
        json.dump(build_chunk_cache_metadata(source_dir, reference_links), f, ensure_ascii=False, indent=2)
    if os.path.exists(faiss_path):
        os.remove(faiss_path)
        print(f"[REBUILD] Removed stale FAISS index: {faiss_path}")
    print(f"[SAVE] Chunks saved to {chunks_path}")

if chunks and isinstance(chunks[0], dict):
    chunks = [c.get("text", "") for c in chunks if c.get("text")]

print(f"Total chunks: {len(chunks)}\n")


# Build / load FAISS index

print(f"[LOAD] Loading SentenceTransformer model '{RETRIEVAL_MODEL}'...")
t0 = time.time()
model = SentenceTransformer(
    RETRIEVAL_MODEL,
    trust_remote_code=True,
    device="cpu",
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
)
print(f"[OK] Model loaded in {time.time() - t0:.1f} seconds.\n")

if os.path.exists(faiss_path):
    print(f"[LOAD] Loading FAISS index from {faiss_path}")
    faiss_index = faiss.read_index(faiss_path)
    expected_dim = model.get_sentence_embedding_dimension()
    if faiss_index.d != expected_dim:
        print(f"[REBUILD] FAISS dim mismatch ({faiss_index.d} != {expected_dim}), rebuilding index.")
        os.remove(faiss_path)
        faiss_index = None
else:
    faiss_index = None

if faiss_index is None:
    print("[BUILD] Creating new FAISS index...")
    batch_size = 8
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Encoding chunks"):
        batch = chunks[i:i + batch_size]
        embs = model.encode(batch, normalize_embeddings=True)
        embeddings.append(embs)
    embeddings = np.vstack(embeddings).astype("float32")

    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, faiss_path)
    del embeddings
    gc.collect()
    print(f"[SAVE] FAISS index saved to {faiss_path}")

# Build per-section queries
print("Building per-section queries from retrieved chunks...")
contexts = [hybrid_retrieve(outline, faiss_index, model, chunks, k=6) for outline in outlines]
queries = [
    f"section number: {i + 1} section title: {sections[i]} section outline: {outlines[i]} section context: {contexts[i]}"
    for i in range(len(sections))
]

# System prompt
system_prompt =  f"""
You are a helpful assistant and expert at making popular YouTube videos.
You will generate the narration script of a video titled: "{video_title}", section by section
The narration is divided into {n_section} sections.

{narration_style}

Guidelines:
- Follow the provided outline and context precisely for each section.
- Maintain natural continuity and smooth transition from the previous section
 -you will receive the following:
 - "section title" = title of the section
  - "section outline" = outline of the section
  - "section context" = context material to generate the section
  - "EARLIER_SECTIONS_SUMMARY" = summary of the sections already generated, to remind you where you are in the story
  - "FULL_OUTLINE_WITH_STATUS"= shows the full outline with flags showing which sections are complete and what is pending

narration_style: 

Respond ONLY with valid JSON in this format:
{{
  "sections": [
    {{
      "section_title": "<same as provided>",
      "narration_text": [
        "Segment 1 (100 to 200 words)",
        "Segment 2 (100 to 200 words)",
        "Segment 3 (optional, 100 to 200 words)"
      ]
    }}
  ]
}}

Each 'narration_text' entry should be a natural continuation of the section, forming 2 to 3 concise narrative segments.
Do not include any text outside this JSON.
"""
# Generate narration JSONs
responses = chat_with_llm(system_prompt, queries, outline_data, my_tkn)

# Parse and save outputs
all_segments = []
final_json = {"video_title": video_title, "n_sections": n_section, "sections": []}

for r in responses:
    try:
        data = json.loads(r)
        for s in data["sections"]:
            final_json["sections"].append({
                "section_title": s["section_title"],
                "narration_text": s["narration_text"]
            })
            all_segments.extend(s["narration_text"])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by model for a section:\n{r}") from e

narration_script = "\n\n".join(all_segments)


json_output_path = os.path.join(json_dir, "narration.json")
with open(json_output_path, "w", encoding="utf-8") as jf:
    json.dump(final_json, jf, ensure_ascii=False, indent=2)
print(f"[OK] Structured JSON saved to {json_output_path}")



