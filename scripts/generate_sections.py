"""generate_sections.py

Module for generating structured section outlines from source materials.

Description:
    Creates detailed section outlines for video by processing source documents,
    extracting relevant information, and using DeepSeek LLM to generate coherent
    section titles and outlines. Intelligently filters source content using
    keyword extraction to ensure outline relevance.

Inputs:
    - source_material/config.json: Project metadata, guidelines, and reference links
    - source_material/*.txt, *.pdf: Local source documents
    - Reference URLs (Wikipedia and web pages)

Outputs:
    - outputs/output_jsons/outline_texts.json: Structured JSON with section titles and outlines

Environment Variables:
    - DEEPSEEK_API_KEY: API key for DeepSeek language model

Configuration:
    - MIN_CHUNK_SIZE: 5000 characters per semantic chunk
    - LLM model: configured DeepSeek model (default: deepseek-v4-flash)

Usage:
    python generate_sections.py <project_name>
    
    Arguments:
        project_name: Project identifier (e.g., 'VikramBetaal')
"""

import os
import fitz  # PyMuPDF
import wikipediaapi
import requests
from bs4 import BeautifulSoup
import numpy as np
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
import re
from sentence_transformers import SentenceTransformer
from console_utils import configure_utf8_output
from deepseek_utils import DEFAULT_DEEPSEEK_MODEL, create_deepseek_chat_completion, get_deepseek_model
from prompt_loader import load_prompt, render_prompt
from text_utils import clean_json_text, clean_text

configure_utf8_output()


MIN_CHUNK_SIZE = 5000
RETRIEVAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------
# Helper: Get project name and paths from config
# --------------------------------------------------------
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

# --------------------------------------------------------
# Load API key from .env
# --------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=ENV_PATH)
my_tkn = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = DEFAULT_DEEPSEEK_MODEL

if not my_tkn:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env file")

# --------------------------------------------------------
# Parse input project argument
# --------------------------------------------------------
project, source_dir, description = load_project_config()
MODEL_NAME = get_deepseek_model(description)
project_root = os.path.dirname(source_dir)
json_dir = os.path.join(project_root, "outputs", "output_jsons")
os.makedirs(json_dir, exist_ok=True)

video_title = description["video_title"]
section_outlines = "\n".join(description["section_outlines"])
n_section = description["n_section"]
reference_links = description.get("reference_links", [])  # may be empty
intro_material = description.get("intro_material", [])
# narration_tone = description.get("narration_tone", "")
# aesthetic_style = description.get("aesthetic_style", "")

print(video_title, "\n", "outline :" , section_outlines, "\n" , "number of sections: ", n_section)


# --------------------------------------------------------
# Utility functions
# --------------------------------------------------------
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

def get_wikipedia_text(title):
    """
    Fetch and return text content from Wikipedia page.
    
    Args:
        title: Wikipedia page title
    
    Returns:
        str: Full page text if found, None otherwise
    """    
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="YourAppName/1.0 (your@email.com)")
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


def get_reference_text(ref):
    """Fetch text from a Wikipedia URL or a general web page URL."""
    if "wikipedia.org/wiki/" in ref:
        title = ref.split("/wiki/")[-1].replace("_", " ")
        return get_wikipedia_text(title)
    return get_webpage_text(ref)

def split_into_chunks(chunks, text, min_chunk_size=MIN_CHUNK_SIZE):
    """Split text into semantic chunks with minimum size requirement.
    
    Args:
        chunks: List to append chunks to
        text: Full text to split
        min_chunk_size: Minimum characters per chunk (default: 5000)
    
    Returns:
        list: Updated chunks list with new chunks appended
    
    Notes:
        - Splits on paragraph boundaries to preserve semantic coherence
        - Ensures each chunk meets minimum size requirement
        - Merges small trailing chunks with previous chunk
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    temp_chunk = ""
    for para in paragraphs:
        if len(temp_chunk) == 0:
            temp_chunk = para
        elif len(temp_chunk) + len(para) < min_chunk_size:
            temp_chunk += " " + para
        else:
            if len(temp_chunk) < min_chunk_size:
                temp_chunk += " " + para
                chunks.append(temp_chunk.strip())
                temp_chunk = ""
            else:
                chunks.append(temp_chunk.strip())
                temp_chunk = para
    if temp_chunk:
        if len(temp_chunk) < min_chunk_size and chunks:
            last_chunk = chunks.pop()
            last_chunk += " " + temp_chunk
            chunks.append(last_chunk.strip())
        else:
            chunks.append(temp_chunk.strip())
    return chunks

# --------------------------------------------------------
# Document processing
# --------------------------------------------------------
def process_all_documents(source_dir, reference_links, intro_material=None):
    """Process local files and online references into semantic chunks."""
    all_chunks = []
    intro_items = intro_material or []

    def add_text_chunks(text, label):
        if not text:
            return
        chunks = []
        docu_chunks = split_into_chunks(chunks, text)
        all_chunks.extend(docu_chunks)
        print("Loaded:", label)

    if intro_items:
        for item in intro_items:
            if isinstance(item, str) and (item.startswith("http://") or item.startswith("https://")):
                fetched = get_reference_text(item)
                add_text_chunks(fetched, item)
                continue

            path = os.path.join(source_dir, item)
            if not os.path.exists(path):
                print(f"Missing intro material: {item}")
                continue

            if item.endswith(".txt"):
                add_text_chunks(read_txt(path), item)
            elif item.endswith(".pdf"):
                add_text_chunks(read_pdf(path), item)
    else:
        for file in os.listdir(source_dir):
            path = os.path.join(source_dir, file)

            if file.endswith(".txt"):
                add_text_chunks(read_txt(path), file)
            elif file.endswith(".pdf"):
                add_text_chunks(read_pdf(path), file)

    if not intro_items:
        for url in reference_links:
            fetched = get_reference_text(url)
            if fetched:
                add_text_chunks(fetched, url)

    # De-duplicate while preserving order
    return list(dict.fromkeys(all_chunks))


def extract_keywords(client, guidelines_text):
    """Extract a compact keyword set from section guidelines."""
    system_prompt = load_prompt("generate_sections_keywords_system.txt")
    response = create_deepseek_chat_completion(
        client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"SECTION_GUIDELINES:\n{guidelines_text}"}
        ],
        stream=False
    )
    keyword_text = clean_text(response.choices[0].message.content.strip())
    print("\n[INFO] Extracted keywords:\n", keyword_text)
    return [kw.strip().lower() for kw in re.split(r"[,\n]", keyword_text) if kw.strip()]


def normalize_section_guidelines(raw_guidelines, expected_count):
    """Normalize the user-provided section_outlines list without relying on labels."""
    cleaned = [str(item).strip() for item in raw_guidelines if str(item).strip()]
    return cleaned


def tokenize(text):
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower()) if len(t) > 2]


def hybrid_retrieve(query, chunks, model, chunk_emb, global_keywords, top_k=8):
    """Keyword + dense retrieval with simple CPU-friendly rerank."""
    if not chunks:
        return []

    q_emb = model.encode([query], normalize_embeddings=True)[0]
    dense_scores = np.dot(chunk_emb, q_emb)
    q_tokens = set(tokenize(query))
    kw_tokens = set(global_keywords)

    candidates = []
    for i, chunk in enumerate(chunks):
        c_tokens = set(tokenize(chunk))
        token_overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
        keyword_overlap = len(kw_tokens & c_tokens) / max(1, len(kw_tokens))
        combined = 0.7 * float(dense_scores[i]) + 0.2 * token_overlap + 0.1 * keyword_overlap
        candidates.append((combined, chunk))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()
    for _, chunk in candidates:
        key = chunk[:200]
        if key in seen:
            continue
        selected.append(chunk)
        seen.add(key)
        if len(selected) >= top_k:
            break
    return selected


def clean_model_value(value):
    """Normalize model output values before saving."""
    if value is None:
        return None
    value = value.strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", value)
        value = re.sub(r"\n?```$", "", value)
        value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def parse_section_response(text):
    """Parse model output robustly for section_title and outline labels."""
    label_matches = list(
        re.finditer(r'(?im)^\s*(section_title|outline)\s*:\s*', text)
    )
    if not label_matches:
        return None, None

    values = {}
    for idx, match in enumerate(label_matches):
        label = match.group(1).lower()
        start = match.end()
        end = label_matches[idx + 1].start() if idx + 1 < len(label_matches) else len(text)
        values[label] = clean_model_value(text[start:end])

    return values.get("section_title"), values.get("outline")


# --------------------------------------------------------
# Main
# --------------------------------------------------------
chunks = process_all_documents(source_dir, reference_links, intro_material)
print(len(chunks))
if not chunks:
    raise ValueError("No source chunks found for section generation.")

string_lengths = [len(s) for s in chunks]
print(f"Mean: {np.mean(string_lengths)}")
print(f"Median: {np.median(string_lengths)}")
print(f"Min: {np.min(string_lengths)}")
print(f"Max: {np.max(string_lengths)}")
print(f"Standard Deviation: {np.std(string_lengths)}")

# --------------------------------------------------------
# LLM call to DeepSeek
# --------------------------------------------------------
client = OpenAI(api_key=my_tkn, base_url="https://api.deepseek.com")
keywords = extract_keywords(client, section_outlines)

print(f"[LOAD] Retrieval model: {RETRIEVAL_MODEL}")
retrieval_model = SentenceTransformer(RETRIEVAL_MODEL, device="cpu")
chunk_embeddings = retrieval_model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
print(f"[OK] Embedded {len(chunks)} chunks for retrieval")

section_guidelines = normalize_section_guidelines(description.get("section_outlines", []), n_section)
print(f"[INFO] Found {len(section_guidelines)} section guidelines")
if not section_guidelines:
    raise ValueError(
        "No section guidelines found in config['section_outlines']. "
        "Provide one outline instruction per desired section."
    )
if len(section_guidelines) != n_section:
    print(f"[WARN] n_section={n_section} but found {len(section_guidelines)} section guideline entries")

outline_data = []
previous_summary = ""

content = load_prompt("generate_sections_outline_system.txt")

for idx, guideline in enumerate(section_guidelines, start=1):
    print(f"\n[SECTION {idx}/{len(section_guidelines)}] Processing")
    print(f"[GUIDELINE] {guideline}")
    retrieved = hybrid_retrieve(guideline, chunks, retrieval_model, chunk_embeddings, keywords, top_k=8)
    print(f"[RETRIEVAL] Selected {len(retrieved)} chunks")
    evidence_text = "\n\n".join(retrieved)
    user_prompt = render_prompt(
        "generate_sections_outline_user.txt",
        video_title=video_title,
        section_number=idx,
        n_section=n_section,
        section_guideline=guideline,
        previous_section_summary=previous_summary if previous_summary else "None",
        evidence=evidence_text,
    )

    print(f"[API] Calling DeepSeek for section {idx}")
    response = create_deepseek_chat_completion(
        client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": user_prompt}
        ],
        stream=False
    )
    print(f"[API] Response received for section {idx}")

    section_text = response.choices[0].message.content
    section_title, outline = parse_section_response(section_text)

    if not section_title or not outline:
        raise ValueError(f"Failed to parse section {idx} from model output:\n{section_text}")

    outline_data.append({"section_title": section_title, "outline": outline})
    previous_summary = f"{section_title}: {outline}"
    print(f"[OK] Parsed section {idx}: {section_title}")
    print(f"[OUTLINE {idx}] {outline}")

# --------------------------------------------------------
# Save as JSON
# --------------------------------------------------------
outline_data = clean_json_text(outline_data)
json_path = os.path.join(json_dir, 'outline_texts.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(outline_data, f, indent=2, ensure_ascii=False)

print(f"[OK] Outline saved as JSON at: {json_path}")
