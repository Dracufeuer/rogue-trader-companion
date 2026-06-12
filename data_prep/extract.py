import os
import re
import json
import argparse
import httpx
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PDF_DIR        = Path("data/pdfs")
FREE_DIR       = Path("data/free_supplements")
EXTRACTED_DIR  = Path("data/extracted")
DATASET_DIR    = Path("finetune/dataset")
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "gemma4:e4b"

# Books and their metadata and their priority in the RAG Layer
BOOK_METADATA = {
    "rogue-trader-core-rulebook": {
        "short_name": "Core Rulebook",
        "priority": 1.0,
        "type": "rules"
    },
    "game-masters-kit": {
        "short_name": "GM's Kit",
        "priority": 0.95,
        "type": "rules"
    },
    "living-errata-v1-4": {
        "short_name": "Living Errata",
        "priority": 0.95,
        "type": "errata"
    },
    "into-the-storm": {
        "short_name": "Into the Storm",
        "priority": 0.85,
        "type": "rules"
    },
    "koronus-bestiary": {
        "short_name": "Koronus Bestiary",
        "priority": 0.85,
        "type": "bestiary"
    },
    "battlefleet-koronus": {
        "short_name": "Battlefleet Koronus",
        "priority": 0.75,
        "type": "rules"
    },
    "edge-of-the-abyss": {
        "short_name": "Edge of the Abyss",
        "priority": 0.70,
        "type": "lore"
    },
    "forsaken-bounty": {
        "short_name": "Forsaken Bounty",
        "priority": 0.65,
        "type": "adventure"
    },
}

# ── Step 1: PDF Text Extraction ────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from a selectable PDF page by page.
    Returns a list of page dicts with text and metadata.

    pymupdf4llm usually gives better reading order than page.get_text("text"),
    but on Rogue Trader stat blocks it can create junk Markdown tables whose
    first row is repeated body text. We keep its narrative text, remove those
    auto tables when PyMuPDF detects real tables, then append clean tables.
    """
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num, page in enumerate(doc):
        tables = page.find_tables().tables
        text = extract_page_markdown(pdf_path, page, page_num, tables)

        pages.append({
            "page": page_num + 1,
            "text": text.strip(),
            "source": pdf_path.stem.lower()
        })

    doc.close()
    print(f"  Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def extract_page_markdown(pdf_path: Path, page, page_num: int, tables: list) -> str:
    """Extract one page as Markdown, repairing tables when possible."""
    try:
        import pymupdf4llm
        text = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
    except Exception:
        text = page.get_text("text")

    text = remove_picture_placeholders(text)
    text = repair_collapsed_visual_tables(text, page_num + 1)

    if tables:
        text = remove_markdown_tables(text)
        table_text = tables_to_markdown(tables)
        if table_text:
            text = f"{text.rstrip()}\n\n{table_text}"

    text = append_raw_text_fallback(text, page)
    return clean_extracted_markdown(text)


def append_raw_text_fallback(markdown_text: str, page) -> str:
    """Append raw page text when Markdown extraction misses profile data."""
    raw_text = page.get_text("text") or ""
    raw_text = normalize_pdf_artifacts(raw_text)

    markers = [
        "WS", "BS", "S", "T", "Ag", "Int", "Per", "WP", "Fel",
        "Strength", "Agility", "Intelligence", "Perception",
        "Willpower", "Fellowship", "Skills:", "Talents:", "Traits:",
    ]

    markdown_has_stats = any(marker in markdown_text for marker in markers)
    raw_has_stats = any(marker in raw_text for marker in markers)

    if raw_has_stats and not markdown_has_stats:
        return f"{markdown_text.rstrip()}\n\n## Raw Extracted Profile Text\n\n{raw_text.strip()}"

    if "Skills:" in raw_text and "Skills:" not in markdown_text:
        return f"{markdown_text.rstrip()}\n\n## Raw Extracted Skill Text\n\n{raw_text.strip()}"

    return markdown_text


def repair_collapsed_visual_tables(text: str, page_number: int) -> str:
    """Repair visual tables that are present in text but not detected by PyMuPDF."""
    text = split_multiline_markdown_table_rows(text)
    text = normalize_markdown_tables(text)
    return text


def split_multiline_markdown_table_rows(text: str) -> str:
    """Split Markdown table rows whose cells contain <br> into separate rows."""
    repaired = []

    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            repaired.append(line)
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not any("<br>" in cell for cell in cells):
            repaired.append(line)
            continue

        split_cells = [re.split(r"\s*<br>\s*", cell) for cell in cells]
        row_count = max(len(parts) for parts in split_cells)

        for row_index in range(row_count):
            row = []
            for parts in split_cells:
                row.append(parts[row_index].strip() if row_index < len(parts) else "")
            repaired.append("|" + "|".join(row) + "|")

    return "\n".join(repaired)


def normalize_markdown_tables(text: str) -> str:
    """Normalize Markdown tables without inventing missing table content."""
    lines = text.splitlines()
    output = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not is_markdown_table_line(line):
            output.append(line)
            index += 1
            continue

        block = []
        while index < len(lines) and is_markdown_table_line(lines[index]):
            block.append(lines[index])
            index += 1

        output.extend(normalize_markdown_table_block(block))

    return "\n".join(output)


def is_markdown_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def normalize_markdown_table_block(block: list[str]) -> list[str]:
    rows = []
    for line in block:
        cells = [clean_table_cell(cell) for cell in line.strip().strip("|").split("|")]
        if any(cells):
            rows.append(cells)

    if not rows:
        return block

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    normalized = []
    for row_index, row in enumerate(rows):
        normalized.append("| " + " | ".join(row) + " |")
        if row_index == 0 and not table_block_has_separator(rows):
            normalized.append("| " + " | ".join(["---"] * width) + " |")

    return normalized


def table_block_has_separator(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in rows[1] if cell.strip())



def remove_picture_placeholders(text: str) -> str:
    """Drop pymupdf4llm image placeholder lines."""
    lines = []
    for line in text.splitlines():
        if "picture" in line.lower() and "intentionally omitted" in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines)


def remove_markdown_tables(text: str) -> str:
    """Remove Markdown table blocks so repaired PyMuPDF tables can replace them."""
    kept = []
    table_block = []

    for line in text.splitlines():
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if is_table_line:
            table_block.append(line)
            continue

        if table_block:
            table_block = []

        kept.append(line)

    return "\n".join(kept)


def tables_to_markdown(tables: list) -> str:
    """Convert PyMuPDF tables to stable Markdown tables."""
    blocks = []

    for table_index, table in enumerate(tables, start=1):
        rows = []
        for row in table.extract():
            cleaned = [clean_table_cell(cell) for cell in row]
            if any(cleaned):
                rows.append(cleaned)

        if not rows:
            continue

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]

        title = None
        if len(rows[0]) > 1 and rows[0][0] and not any(rows[0][1:]):
            title = rows.pop(0)[0]

        if not rows:
            continue

        header = rows[0]
        body = rows[1:]

        block = []
        if title:
            block.append(f"**{title}**")
            block.append("")
        elif len(tables) > 1:
            block.append(f"**Table {table_index}**")
            block.append("")

        block.append("| " + " | ".join(header) + " |")
        block.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in body:
            block.append("| " + " | ".join(row) + " |")

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


def clean_table_cell(cell) -> str:
    """Normalize a table cell for Markdown output."""
    if cell is None:
        return ""
    text = str(cell).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("|", "/")
    return normalize_pdf_artifacts(text)


def clean_extracted_markdown(text: str) -> str:
    """Light cleanup for extracted Markdown."""
    text = text.replace("\r\n", "\n")
    text = normalize_pdf_artifacts(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def normalize_pdf_artifacts(text: str) -> str:
    """Repair common separated-ligature artifacts from these PDFs."""
    ligatures = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
        "ï¬€": "ff",
        "ï¬": "fi",
        "ï¬‚": "fl",
        "ï¬ƒ": "ffi",
        "ï¬„": "ffl",
        "ï¬…": "ft",
        "ï¬†": "st",
    }
    for bad, good in ligatures.items():
        text = text.replace(bad, good)

    # Some PDFs extract words like "first" and "floor" as "fi rst" / "fl oor".
    # Repair only the space after a ligature pair, preserving normal word spaces.
    text = re.sub(r"\b(fi|fl)\s+([a-z])", r"\1\2", text)
    text = re.sub(r"([A-Za-z])(fi|fl)\s+([a-z])", r"\1\2\3", text)

    replacements = {
        "Profi le": "Profile",
        "Profil e": "Profile",
        "Prof le": "Profile",
        "profi le": "profile",
        "Ord enProfile": "Orden Profile",
        "fi rst": "first",
        "fi fth": "fifth",
        "fi ght": "fight",
        "fi ghts": "fights",
        "fi ring": "firing",
        "fi re": "fire",
        "fi eld": "field",
        "fi elds": "fields",
        "fi nd": "find",
        "fi nds": "finds",
        "fi nal": "final",
        "fi ll": "fill",
        "fi lthy": "filthy",
        "fi repower": "firepower",
        "fl oor": "floor",
        "fl ashes": "flashes",
        "fl ow": "flow",
        "fl ows": "flows",
        "fl ee": "flee",
        "fl uid": "fluid",
        "fl uidly": "fluidly",
        "offl ine": "offline",
        "briefl y": "briefly",
        "diffi cult": "difficult",
        "horrifi c": "horrific",
        "identifi es": "identifies",
        "terrifi ed": "terrified",
        "modifi ed": "modified",
        "inf ormation": "information",
        "Ordinaty": "Ordinary",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def extract_all_pdfs() -> dict[str, list[dict]]:
    """Extract text from all PDFs in both PDF directories."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    all_books = {}

    for pdf_dir in [PDF_DIR, FREE_DIR]:
        if not pdf_dir.exists():
            print(f"Directory not found, skipping: {pdf_dir}")
            continue

        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            book_key = pdf_path.stem.lower().replace(" ", "-")
            print(f"Extracting: {pdf_path.name}")

            pages = extract_pdf(pdf_path)
            all_books[book_key] = pages

            # Save raw extracted text for inspection
            out_path = EXTRACTED_DIR / f"{book_key}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                for page in pages:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"PAGE {page['page']}\n")
                    f.write(f"{'='*60}\n")
                    f.write(page["text"])
                    f.write("\n")

            print(f"  Saved to {out_path}")

    return all_books


# ── Step 2: Text Cleaning ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    Removes headers, footers, page numbers, and OCR artifacts.
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        # Skip very short lines (likely headers/footers/page numbers)
        if len(line) < 4:
            continue

        # Skip lines that are just numbers (page numbers)
        if re.match(r"^\d+$", line):
            continue

        # Skip common RPG book headers/footers
        skip_patterns = [
            r"^rogue trader$",
            r"^warhammer 40",
            r"^fantasy flight",
            r"^chapter \w+$",
            r"^\d+\s*$",
        ]
        if any(re.match(p, line.lower()) for p in skip_patterns):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ── Step 3: Section Detection ──────────────────────────────────────────────────

def detect_sections(pages: list[dict], book_type: str) -> list[dict]:
    """
    Split pages into meaningful sections based on content type.
    Different books need different splitting strategies.
    """
    sections = []

    if book_type == "bestiary":
        # Bestiary: split by enemy entry
        sections = split_bestiary_entries(pages)
    elif book_type == "rules":
        # Rules: split by section headers
        sections = split_by_headers(pages)
    elif book_type == "lore":
        # Lore: split by topic chunks
        sections = split_by_topic(pages)
    else:
        # Default: split by page groups
        sections = split_by_pages(pages, chunk_size=3)

    return sections


def split_bestiary_entries(pages: list[dict]) -> list[dict]:
    """
    Split a bestiary into individual enemy entries.
    Detects stat block patterns (WS/BS/S/T characteristic lines).
    """
    sections = []
    current_entry = []
    current_name = None

    stat_pattern = re.compile(
        r"WS\s+BS\s+S\s+T|"
        r"\d{2}\s+\d{2}\s+\d{2}\s+\d{2}"
    )

    for page in pages:
        lines = page["text"].split("\n")
        for line in lines:
            # Detect what looks like an enemy name (short, title-cased, before stats)
            if (len(line) < 40 and
                line.istitle() and
                not stat_pattern.search(line) and
                len(current_entry) > 5):
                # Save previous entry
                if current_name and current_entry:
                    sections.append({
                        "title": current_name,
                        "text": "\n".join(current_entry),
                        "type": "bestiary_entry",
                        "source": page["source"]
                    })
                current_name = line
                current_entry = [line]
            else:
                current_entry.append(line)

    # Don't forget the last entry
    if current_name and current_entry:
        sections.append({
            "title": current_name,
            "text": "\n".join(current_entry),
            "type": "bestiary_entry",
            "source": pages[-1]["source"] if pages else "unknown"
        })

    return sections


def split_by_headers(pages: list[dict]) -> list[dict]:
    """
    Split rules text by section headers.
    Detects ALL CAPS headers and numbered sections.
    """
    sections = []
    current_section = []
    current_title = "Introduction"

    header_pattern = re.compile(r"^[A-Z][A-Z\s]{4,}$|^\d+\.\d*\s+[A-Z]")

    for page in pages:
        lines = page["text"].split("\n")
        for line in lines:
            line = line.strip()
            if header_pattern.match(line) and len(line) < 60:
                # Save previous section if substantial
                if len(current_section) > 10:
                    sections.append({
                        "title": current_title,
                        "text": "\n".join(current_section),
                        "type": "rules_section",
                        "source": page["source"]
                    })
                current_title = line
                current_section = [line]
            else:
                current_section.append(line)

    # Save last section
    if current_section:
        sections.append({
            "title": current_title,
            "text": "\n".join(current_section),
            "type": "rules_section",
            "source": pages[-1]["source"] if pages else "unknown"
        })

    return sections


def split_by_topic(pages: list[dict]) -> list[dict]:
    """Group pages into topic chunks for lore books."""
    sections = []
    chunk_size = 4

    for i in range(0, len(pages), chunk_size):
        chunk = pages[i:i + chunk_size]
        text = "\n\n".join(p["text"] for p in chunk)
        sections.append({
            "title": f"Pages {chunk[0]['page']}-{chunk[-1]['page']}",
            "text": text,
            "type": "lore_section",
            "source": chunk[0]["source"]
        })

    return sections


def split_by_pages(pages: list[dict], chunk_size: int = 3) -> list[dict]:
    """Default: group pages into chunks."""
    sections = []
    for i in range(0, len(pages), chunk_size):
        chunk = pages[i:i + chunk_size]
        text = "\n\n".join(p["text"] for p in chunk)
        sections.append({
            "title": f"Pages {chunk[0]['page']}-{chunk[-1]['page']}",
            "text": text,
            "type": "general",
            "source": chunk[0]["source"]
        })
    return sections


# ── Step 4: Q&A Generation ─────────────────────────────────────────────────────

def generate_qa_from_section(section: dict) -> list[dict]:
    """
    Use the local Ollama model to generate Q&A pairs from a text section.
    Returns a list of {"prompt": ..., "completion": ...} dicts.
    """
    section_type = section.get("type", "general")

    # Choose the right generation prompt based on section type
    if section_type == "bestiary_entry":
        system_instruction = """You are generating training data for a Rogue Trader RPG assistant.
Given this enemy stat block, generate 8 question-answer pairs that a DM would find useful.
Focus on: stat lookups, special abilities, tactical advice, and how to use this enemy effectively.
Output ONLY a JSON array with this exact format, nothing else:
[
  {"prompt": "question here", "completion": "answer here"},
  {"prompt": "question here", "completion": "answer here"},
  {"prompt": "question here", "completion": "answer here"}
]"""

    elif section_type == "rules_section":
        system_instruction = """You are generating training data for a Rogue Trader RPG assistant.
Given this rules section, generate 8 question-answer pairs a DM would ask mid-session.
Focus on: how mechanics work, what modifiers apply, edge cases, and practical application.
Answers should be conversational and concise — a DM needs fast answers at the table.
Output ONLY a JSON array with this exact format, nothing else:
[
  {"prompt": "question here", "completion": "answer here"},
  {"prompt": "question here", "completion": "answer here"},
  {"prompt": "question here", "completion": "answer here"}
]"""

    elif section_type == "lore_section":
        system_instruction = """You are generating training data for a Rogue Trader RPG assistant.
Given this lore passage, generate 8 questions about setting, factions, locations or characters.
Answers should be flavourful and atmospheric, written in the tone of the 40K universe.
Output ONLY a JSON array with this exact format, nothing else:
[
  {"prompt": "question here", "completion": "answer here"},
  {"prompt": "question here", "completion": "answer here"},
  {"prompt": "question here", "completion": "answer here"}
]"""

    else:
        system_instruction = """You are generating training data for a Rogue Trader RPG assistant.
Given this text, generate 8 useful question-answer pairs for a DM running a Rogue Trader session.
Output ONLY a JSON array with this exact format, nothing else:
[
  {"prompt": "question here", "completion": "answer here"},
  {"prompt": "question here", "completion": "answer here"},
  {"prompt": "question here", "completion": "answer here"}
]"""

    # Truncate very long sections to avoid overwhelming the model
    text = section["text"][:3000]

    full_prompt = f"{system_instruction}\n\nTEXT:\n{text}\n\nJSON OUTPUT:"

    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 2500
                }
            },
            timeout=120.0
        )

        raw = response.json()["response"].strip()

        # Strip markdown code fences if model adds them
        raw = re.sub(r"```json|```", "", raw).strip()

        # Find the JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            print(f"  Warning: No JSON array found in response for section '{section['title']}'")
            return []

        pairs = json.loads(match.group())

        # Validate structure
        valid = []
        for pair in pairs:
            if isinstance(pair, dict) and "prompt" in pair and "completion" in pair:
                if len(pair["prompt"]) > 10 and len(pair["completion"]) > 20:
                    valid.append(pair)

        return valid

    except httpx.ConnectError:
        print("  Error: Cannot connect to Ollama. Is it running? Run: ollama serve")
        return []
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse error for section '{section['title']}': {e}")
        return []
    except Exception as e:
        print(f"  Error generating Q&A for '{section['title']}': {e}")
        return []


# ── Step 5: NPC Dataset from Bestiary ─────────────────────────────────────────

def generate_npc_dataset(bestiary_sections: list[dict]) -> list[dict]:
    """
    Generate NPC generation training pairs from bestiary entries.
    Format: role description → full stat block.
    """
    npc_pairs = []

    for section in bestiary_sections:
        if section.get("type") != "bestiary_entry":
            continue

        name = section["title"]
        text = section["text"]

        prompt = f"""Generate a training example for an NPC generator.
Given this stat block for '{name}', create a natural role description that a DM might type
to request this kind of enemy.

STAT BLOCK:
{text[:2000]}

Output ONLY a JSON object with this format, nothing else:
{{"prompt": "Generate a [threat level] [faction] NPC. Role: [natural description]", "completion": "[full stat block formatted cleanly]"}}"""

        try:
            response = httpx.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 800}
                },
                timeout=120.0
            )

            raw = response.json()["response"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                pair = json.loads(match.group())
                if "prompt" in pair and "completion" in pair:
                    npc_pairs.append(pair)

        except Exception as e:
            print(f"  Error generating NPC pair for '{name}': {e}")
            continue

    return npc_pairs


# ── Step 6: Save Dataset ───────────────────────────────────────────────────────

def save_jsonl(pairs: list[dict], output_path: Path):
    """Save a list of pairs to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to existing file if it exists
    mode = "a" if output_path.exists() else "w"

    with open(output_path, mode, encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"  Saved {len(pairs)} pairs to {output_path}")


def count_dataset():
    """Print current dataset statistics."""
    print("\n📊 Dataset Statistics:")
    total = 0
    for jsonl_file in sorted(DATASET_DIR.glob("*.jsonl")):
        count = sum(1 for _ in open(jsonl_file, encoding="utf-8"))
        total += count
        print(f"  {jsonl_file.name}: {count} examples")
    print(f"  Total: {total} examples")
    print(f"  Target: 500-800 examples")
    if total < 500:
        print(f"  Still need: {500 - total} more examples")


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_extraction():
    """Step 1-2: Extract and save raw text from PDFs."""
    print("\n🔍 Extracting text from PDFs...")
    all_books = extract_all_pdfs()

    if not all_books:
        print("No PDFs found. Add your books to data/pdfs/ first.")
        return {}

    print(f"\n✅ Extracted {len(all_books)} books")
    return all_books


def run_qa_generation(all_books: dict):
    """Step 3-5: Generate Q&A pairs from extracted text."""
    print("\n🤖 Generating Q&A training pairs...")
    print("This uses your local Ollama model — make sure it's running.\n")

    rules_pairs = []
    lore_pairs = []
    npc_pairs = []

    for book_key, pages in all_books.items():
        meta = BOOK_METADATA.get(book_key, {"type": "general", "short_name": book_key})
        book_type = meta["type"]

        print(f"Processing: {meta['short_name']} ({len(pages)} pages, type: {book_type})")

        # Detect and split into sections
        sections = detect_sections(pages, book_type)
        print(f"  Found {len(sections)} sections")

        # Generate Q&A for each section
        for i, section in enumerate(sections):
            print(f"  Section {i+1}/{len(sections)}: {section['title'][:50]}...")

            if book_type == "bestiary":
                # Generate NPC training pairs from bestiary
                pairs = generate_npc_dataset([section])
                npc_pairs.extend(pairs)

                # Also generate Q&A about the enemy
                qa_pairs = generate_qa_from_section(section)
                rules_pairs.extend(qa_pairs)

            elif book_type in ["rules", "errata"]:
                qa_pairs = generate_qa_from_section(section)
                rules_pairs.extend(qa_pairs)

            elif book_type == "lore":
                qa_pairs = generate_qa_from_section(section)
                lore_pairs.extend(qa_pairs)

            else:
                qa_pairs = generate_qa_from_section(section)
                rules_pairs.extend(qa_pairs)

    # Save all datasets
    print("\n💾 Saving datasets...")

    if rules_pairs:
        save_jsonl(rules_pairs, DATASET_DIR / "rules_qa.jsonl")

    if lore_pairs:
        save_jsonl(lore_pairs, DATASET_DIR / "lore_qa.jsonl")

    if npc_pairs:
        save_jsonl(npc_pairs, DATASET_DIR / "npc_generation.jsonl")

    count_dataset()


def main():
    parser = argparse.ArgumentParser(description="Rogue Trader PDF Data Extraction Pipeline")
    parser.add_argument("--extract-only", action="store_true", help="Only extract text, skip Q&A generation")
    parser.add_argument("--qa-only", action="store_true", help="Only generate Q&A from already extracted text")
    args = parser.parse_args()

    print("=" * 60)
    print("  Rogue Trader Companion — Data Preparation Pipeline")
    print("=" * 60)

    if args.qa_only:
        # Load from already extracted text files
        all_books = {}
        if EXTRACTED_DIR.exists():
            print("Loading previously extracted text...")
            # Re-extract from saved text files
            for txt_file in EXTRACTED_DIR.glob("*.txt"):
                book_key = txt_file.stem
                with open(txt_file, encoding="utf-8") as f:
                    content = f.read()
                # Reconstruct minimal page structure
                all_books[book_key] = [{"page": 1, "text": content, "source": book_key}]
        run_qa_generation(all_books)

    elif args.extract_only:
        run_extraction()

    else:
        # Full pipeline
        all_books = run_extraction()
        if all_books:
            run_qa_generation(all_books)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
