# ULTIMATE_fixed4.py
# Fast streaming builder for term_bank_*.json with 8 MB cap.
# Adds: Indonesian example key-highlighting with lime color and non-spammy warnings.
# Base derived from ULTIMATE_fixed2.py (structure, writers, and schema-safe nodes).
#
# Behavior:
# - In Indonesian example ("contoh"), occurrences of the entry key are wrapped in a <span>
#   with CSS color: color-mix(in srgb, lime, var(--text-color, var(--fg, #333))).
# - Word-boundary, case-insensitive matches; all occurrences are colored.
# - If the key is absent in the Indonesian example, we keep the sentence as-is and emit
#   at most one warning per key (global), capped to _MISS_WARN_LIMIT for the whole run.
#
# Default: _TERM_BANK_GENERATE_LIMIT = 1000

from __future__ import annotations
import csv, json, re, sys
from pathlib import Path
from urllib.parse import quote as url_encode

# =========================
# Config
# =========================
_TERM_BANK_GENERATE_LIMIT = 1000        # write at most this many term_bank_*.json files
_MAX_JSON_BYTES = 8 * 1024 * 1024       # 8 MB per file

PREPROCESS_FILE = "kbbi_preprocess_ULTIMATE_audited.csv"
GOOGLECONTOH_FILE = "googlecontohKBBI2.4_ULTIMATE.csv"
HYPERLINKS_FILE = "hyperlinks_meanings_ULTIMATE.csv"
FREQ_FILE = "c4_frequency_rank_ULTIMATE.csv"
TAGMAP_FILE = "yomitan_tags_map_ULTIMATE.csv"
LITE_FILE = "LITE_KBBI_ULTIMATE_v2.4.csv"

FIXED_VALUE_AT_4 = 69
EMPTY_STR_AT_3 = ""
EMPTY_STR_AT_7 = ""

POS_EN = {
    "Nomina": "Noun",
    "Verba": "Verb",
    "Adjektiva": "Adjective",
    "Adverbia": "Adverb",
    "Pronomina": "Pronoun",
    "Numeralia": "Numeral",
    "Partikel": "Particle",
}

FOOTER_KBBI = "https://kbbi.web.id/{key}"
FOOTER_TATOEBA = "https://tatoeba.org/en/sentences/search?from=ind&query={key}&to="

BOTTOM_LABELS = [
    ("Kata Dasar",   "kata_dasar"),
    ("Kata Turunan", "kata_turunan"),
    ("Gabungan Kata","gabungan_kata"),
    ("Varian",       "varian"),
]

# =========================
# Logging
# =========================
def warn(msg: str): print(f"[WARN] {msg}", file=sys.stderr)
def info(msg: str): print(f"[INFO] {msg}", file=sys.stderr)

# Limit console noise for "key not in contoh" messages
_MISS_WARN_LIMIT = 25
_MISS_WARN_COUNT = 0
_MISS_SUPPRESSED = 0
_MISS_WARNED_KEYS: set[str] = set()

def warn_example_miss_once(key: str):
    """Warn at most once per key, with a global cap; count suppressed for summary."""
    global _MISS_WARN_COUNT, _MISS_SUPPRESSED
    kl = key.lower()
    if kl in _MISS_WARNED_KEYS:
        _MISS_SUPPRESSED += 1
        return
    if _MISS_WARN_COUNT >= _MISS_WARN_LIMIT:
        _MISS_SUPPRESSED += 1
        return
    warn(f"Example highlight: key={key!r} not found in Indonesian example; leaving uncolored.")
    _MISS_WARNED_KEYS.add(kl)
    _MISS_WARN_COUNT += 1

# =========================
# Helpers
# =========================
_CIRCLED = {i: chr(0x2460 + (i - 1)) for i in range(1, 21)}  # 1..20
def make_circle_number(n: int) -> str:
    if 1 <= n <= 20: return _CIRCLED[n]
    warn(f"Sense number {n} > 20; falling back to '({n})'.")
    return f"({n})"

def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]

def load_data(workdir: Path):
    preprocess = read_csv(workdir / PREPROCESS_FILE)
    examples   = read_csv(workdir / GOOGLECONTOH_FILE)
    hyperlinks = read_csv(workdir / HYPERLINKS_FILE)
    freq       = read_csv(workdir / FREQ_FILE)
    tagmap     = read_csv(workdir / TAGMAP_FILE)
    lite       = read_csv(workdir / LITE_FILE)
    return preprocess, examples, hyperlinks, freq, tagmap, lite

def index_preprocess(pre_rows: list[dict]):
    by_key, keys_lower = {}, set()
    for r in pre_rows:
        k = r["key"]
        by_key[k] = r
        keys_lower.add(k.lower())
    return by_key, keys_lower

def index_lite(lite_rows: list[dict]):
    by_key, key_sense = {}, set()
    for r in lite_rows:
        k = r["key"]
        by_key.setdefault(k, []).append(r)
        nomor = (r.get("nomor") or "").strip()
        if nomor.isdigit():
            key_sense.add((k.lower(), int(nomor)))
    return by_key, key_sense

def index_hyperlinks(h_rows: list[dict]):
    by_row = {}
    for r in h_rows:
        rid = (r.get("row_index") or "").strip()
        s = (r.get("hyperlinks") or "").strip()
        by_row[rid] = [t.strip() for t in s.split(" | ")] if s else []
    return by_row

def index_examples(ex_rows: list[dict]):
    by_row = {}
    for r in ex_rows:
        rid = (r.get("row_index") or "").strip()
        contoh = (r.get("contoh") or "").strip()
        google = (r.get("googlecontoh") or "").strip()
        if contoh or google:
            by_row[rid] = (contoh, google)
    return by_row

def index_freq(freq_rows: list[dict]):
    by_key = {}
    for r in freq_rows:
        k = r["key"]
        rs = (r.get("rank") or "").strip()
        if rs.isdigit():
            by_key[k] = int(rs)
    return by_key

def index_tags(tag_rows: list[dict]):
    by_tag = {}
    for r in tag_rows:
        t = r["tag"]
        by_tag[t] = {
            "rename": r.get("tag_rename") or t,
            "type": r.get("type") or "default",
            "color": r.get("color") or "#626273",
        }
    return by_tag

_SENSE_TAIL_RE = re.compile(r"\s*\((\d+)\)\s*$")
def split_token_and_sense(original: str):
    tok = original.strip()
    m = _SENSE_TAIL_RE.search(tok)
    if m:
        n = int(m.group(1))
        key_only = _SENSE_TAIL_RE.sub("", tok).strip()
        return key_only.lower(), n, key_only, f"({n})"
    return tok.lower(), None, tok, ""

def display_with_circled(key_display: str, sense_text: str):
    if sense_text:
        m = re.match(r"\((\d+)\)", sense_text)
        if m: return key_display + make_circle_number(int(m.group(1)))
    return key_display

def build_internal_href_for_key(key_display_original: str):
    return f"?query={url_encode(key_display_original)}"

# =========================
# Styles (safe subset)
# =========================
def style_pos_badge(color_hex: str):
    return {
        "fontSize": "0.8em",
        "fontWeight": "bold",
        "padding": "0.2em 0.3em",
        "wordBreak": "keep-all",
        "borderRadius": "0.3em",
        "verticalAlign": "text-bottom",
        "backgroundColor": color_hex,
        "color": "white",
        "cursor": "help",
        "marginRight": "0.25em",
    }

def style_tag_badge(color_hex: str):
    return {
        "fontSize": "0.8em",
        "fontWeight": "bold",
        "padding": "0.2em 0.3em",
        "wordBreak": "keep-all",
        "borderRadius": "0.3em",
        "verticalAlign": "text-bottom",
        "backgroundColor": color_hex,
        "color": "white",
        "cursor": "help",
        "marginRight": "0.25em",
    }

def style_arrow_li():
    # attach style directly to the LI (no nested span)
    return {"fontSize": "1em", "marginRight": "0.5rem", "color": "#000001"}

def style_top_ul():
    return {"listStyleType": "\"＊\""}

def style_sense_li(circle: str):
    return {"listStyleType": f"\"{circle}\"", "paddingLeft": "0.25em"}

def style_footer():
    return {"fontSize": "0.7em", "textAlign": "right"}

def style_bottom_value_span():
    return {"fontSize": "1em", "marginRight": "0.5rem", "color": "#000001", "marginLeft": "0.35em"}

def style_example_id(): return {"fontSize": "1em"}
def style_example_en(): return {"fontSize": "0.7em"}

# Highlight style for Indonesian key
def style_example_highlight():
    return {"color": "color-mix(in srgb, lime, var(--text-color, var(--fg, #333)))"}

# =========================
# JSON node builders
# =========================
def badge_span(label: str, color_hex: str, title: str | None = None):
    node = {"tag": "span", "style": style_pos_badge(color_hex), "content": label}
    if title: node["title"] = title
    return node

def default_tag_span(label: str, color_hex: str):
    return {"tag": "span", "style": style_tag_badge(color_hex), "content": label}

def redirect_li(link_nodes: list):
    # Render as a styled LI with content list: ["→ ", <a>, ", ", <a>, ...]
    return {"tag": "li", "style": style_arrow_li(), "content": ["→ "] + link_nodes}

def _highlight_indonesian_sentence_nodes(sentence: str, highlight_key: str):
    """
    Split the sentence and wrap all case-insensitive, word-boundary matches of highlight_key
    into a styled <span>. Returns either the original string (no match) or a list of nodes.
    """
    if not sentence or not highlight_key:
        return sentence
    # Build a Unicode word-boundary pattern
    pat = re.compile(rf"(?i)\b({re.escape(highlight_key)})\b")
    last = 0
    nodes: list = []
    matched = False
    for m in pat.finditer(sentence):
        s, e = m.start(1), m.end(1)
        if s > last:
            nodes.append(sentence[last:s])
        nodes.append({"tag": "span", "style": style_example_highlight(), "content": sentence[s:e]})
        last = e
        matched = True
    if not matched:
        return sentence
    if last < len(sentence):
        nodes.append(sentence[last:])
    return nodes

def example_block(contoh: str, googlecontoh: str, highlight_key: str):
    # Fancy "bubble" examples, matching your working sample (3 nested divs with border + soft bg)
    inner = []
    if contoh:
        hilite = _highlight_indonesian_sentence_nodes(contoh, highlight_key)
        if isinstance(hilite, list):
            inner.append({"tag": "div", "style": {"fontSize": "1em"}, "content": hilite})
        else:
            # no match; keep as-is and warn once (non-spammy)
            warn_example_miss_once(highlight_key)
            inner.append({"tag": "div", "style": {"fontSize": "1em"}, "content": contoh})
    if googlecontoh:
        inner.append({"tag": "div", "style": {"fontSize": "0.7em"}, "content": googlecontoh})
    return {
        "tag": "div",
        "style": {"marginLeft": "0.5em"},
        "content": {
            "tag": "div",
            "content": {
                "tag": "div",
                "style": {
                    "borderStyle": "none none none solid",
                    "padding": "0.5rem",
                    "borderRadius": "0.4rem",
                    "borderWidth": "calc(3em / var(--font-size-no-units, 14))",
                    "marginTop": "0.5rem",
                    "marginBottom": "0.5rem",
                    "borderColor": "var(--text-color, var(--fg, #333))",
                    "backgroundColor": "color-mix(in srgb, var(--text-color, var(--fg, #333)) 5%, transparent)"
                },
                "content": inner
            }
        }
    }

def footer_block(key: str):
    href_kbbi = FOOTER_KBBI.format(key=url_encode(key))
    href_tato = FOOTER_TATOEBA.format(key=url_encode(key))
    return {
        "tag": "div",
        "style": style_footer(),
        "content": [
            {"tag": "a", "href": href_kbbi, "content": "KBBI"},
            " | ",
            {"tag": "a", "href": href_tato, "content": "Tatoeba"},
        ],
    }

def bottom_line_item(label_visible: str, values_nodes: list):
    # Each bottom field is its own LI line
    label_span = {
        "tag": "span",
        "style": {
            "fontSize": "0.8em",
            "fontWeight": "bold",
            "padding": "0.2em 0.3em",
            "wordBreak": "keep-all",
            "borderRadius": "0.3em",
            "verticalAlign": "text-bottom",
            "backgroundColor": "#0000EE",
            "color": "white",
            "cursor": "help",
            "marginRight": "0.25em",
        },
        "content": label_visible,
    }
    value_span = {"tag": "span", "style": style_bottom_value_span(), "content": values_nodes}
    return {"tag": "li", "content": [label_span, value_span]}

def structured_root(children: list):
    return {"type": "structured-content", "content": children}

# =========================
# Rendering utils
# =========================
def build_anchor_or_text(token_original: str, keys_lower: set[str], key_sense: set[tuple[str,int]]):
    k_lower, sense, disp_key, disp_sense = split_token_and_sense(token_original)
    visible_key = disp_key
    visible_full = display_with_circled(visible_key, disp_sense)

    if sense is not None and (k_lower, sense) in key_sense:
        return {"tag": "a", "href": build_internal_href_for_key(disp_key), "content": visible_full}
    elif k_lower in keys_lower:
        parts = [{"tag": "a", "href": build_internal_href_for_key(disp_key), "content": visible_key}]
        if disp_sense:
            parts.append(make_circle_number(int(disp_sense.strip("()"))))
        return parts
    else:
        return visible_full

def join_with_commas(node_list: list):
    out = []
    for i, n in enumerate(node_list):
        if i > 0: out.append(", ")
        out.append(n)
    return out

def prefix_if_narrower(row_pos: list[str], block_pos_sig: list[str], s: str):
    set_row, set_blk = set(row_pos), set(block_pos_sig)
    if set_row and set_row < set_blk:
        if len(set_row) == 1: p = next(iter(set_row)).lower()
        else: p = "/".join(sorted(t.lower() for t in set_row))
        return f"{p} {s}"
    return s

# =========================
# Build one entry
# =========================
def build_entry_array_for_key(
    key: str,
    pre: dict,
    lite_rows: list[dict] | None,
    hyperlinks_by_row: dict[str, list[str]],
    examples_by_row: dict[str, tuple[str,str]],
    freq_by_key: dict[str,int],
    tagmap: dict[str, dict],
    keys_lower: set[str],
    key_sense: set[tuple[str,int]],
):
    reading = pre.get("reading") or ""
    number_index = int(pre.get("number_index") or 0)

    star = ""
    rank = freq_by_key.get(key)
    if rank is not None and rank <= 20000:
        star = "★"

    rows = lite_rows or []
    parsed_rows = []
    pos_first_color = {}

    for r in rows:
        row_index = (r.get("row_index") or "").strip()
        kelas = (r.get("kelas") or "").strip()
        nomor = (r.get("nomor") or "").strip()
        googletranslate = (r.get("googletranslate") or "").strip()

        pos_tags, pos_colors, nonpos_tags, nonpos_colors = [], [], [], []
        tags = [t.strip() for t in kelas.split("; ")] if kelas else []
        for t in tags:
            m = tagmap.get(t)
            if not m: m = {"rename": t, "type": "default", "color": "#626273"}
            if (m.get("type") or "default") == "partOfSpeech":
                pos_tags.append(m["rename"]); c = m["color"]; pos_colors.append(c)
                if m["rename"] not in pos_first_color: pos_first_color[m["rename"]] = c
            else:
                nonpos_tags.append(m["rename"]); nonpos_colors.append(m["color"])

        sense = int(nomor) if nomor.isdigit() else None

        # Redirect overrides gloss
        redir_tokens = hyperlinks_by_row.get(row_index, [])
        if redir_tokens:
            link_nodes = []
            for tok in redir_tokens:
                node = build_anchor_or_text(tok, keys_lower, key_sense)
                if isinstance(node, list): link_nodes.extend(node)
                else: link_nodes.append(node)
            def_node = redirect_li(join_with_commas(link_nodes))
        else:
            if not googletranslate:
                googletranslate = "poor data set, pls report to author"
                warn(f"No redirect or gloss for key={key!r}, row_index={row_index}; inserting placeholder.")
            def_node = {"tag": "li", "content": googletranslate}

        ex = examples_by_row.get(row_index)

        parsed_rows.append({
            "row_index": row_index,
            "pos_tags": pos_tags, "pos_colors": pos_colors,
            "nonpos_tags": nonpos_tags, "nonpos_colors": nonpos_colors,
            "sense": sense, "def_node": def_node, "example": ex,
        })

    # Collapsed case: multiple defs, single tag-group, and no senses anywhere
    collapsed = False
    if parsed_rows:
        all_none = all(pr["sense"] is None for pr in parsed_rows)
        if all_none:
            def tag_sig(pr): return tuple(pr["pos_tags"]) + ("||SEP||",) + tuple(pr["nonpos_tags"])
            first_sig = tag_sig(parsed_rows[0])
            collapsed = all(tag_sig(pr) == first_sig for pr in parsed_rows)

    top_children = []
    top_ul = {"tag": "ul", "style": style_top_ul(), "content": []}

    if not parsed_rows:
        top_ul["content"].append({
            "tag": "li",
            "content": [{"tag": "ul", "content": [{"tag": "li", "content": "poor data set, pls report to author"}]}]
        })
    elif collapsed:
        first = parsed_rows[0]
        header_nodes = []
        for i, pos in enumerate(first["pos_tags"]):
            color = first["pos_colors"][i] if i < len(first["pos_colors"]) else pos_first_color.get(pos, "#626273")
            header_nodes.append(badge_span(pos, color, POS_EN.get(pos, pos)))
        for i, tag in enumerate(first["nonpos_tags"]):
            color = first["nonpos_colors"][i] if i < len(first["nonpos_colors"]) else "#626273"
            header_nodes.append(default_tag_span(tag, color))

        # Build the inner content (definitions + example blocks), *without* headers
        content_nodes, ul_items = [], []
        def flush_ul():
            nonlocal ul_items, content_nodes
            if ul_items:
                content_nodes.append({"tag": "ul", "content": ul_items})
                ul_items = []

        for pr in parsed_rows:
            ul_items.append(pr["def_node"])
            if pr["example"]:
                flush_ul()
                contoh, googlecontoh = pr["example"]
                content_nodes.append(example_block(contoh, googlecontoh, key))
        flush_ul()

        # Now wrap into a single-sense OL with circled 1
        block_content = []
        if header_nodes:
            block_content.extend(header_nodes)
        block_content.append({
            "tag": "ol",
            "content": [
                {
                    "tag": "li",
                    "style": style_sense_li(make_circle_number(1)),
                    "content": content_nodes
                }
            ]
        })

        top_ul["content"].append({"tag": "li", "content": block_content})
    else:
        canon_by_set: dict[frozenset[str], list[str]] = {}
        blocks: dict[tuple[str,...], dict[object, list[dict]]] = {}
        block_order: list[tuple[str,...]] = []

        for pr in parsed_rows:
            pos_sig = pr["pos_tags"]
            pos_set = frozenset(pos_sig)
            if pos_set not in canon_by_set:
                canon_by_set[pos_set] = list(pos_sig)  # first seen order
            block_sig = tuple(canon_by_set[pos_set])
            if block_sig not in blocks:
                blocks[block_sig] = {}
                block_order.append(block_sig)
            s_key = pr["sense"] if pr["sense"] is not None else "__NOSENSE__"
            blocks[block_sig].setdefault(s_key, []).append(pr)

        for block_sig in block_order:
            block_content = []
            for pos in block_sig:
                block_content.append(badge_span(pos, pos_first_color.get(pos, "#626273"), POS_EN.get(pos, pos)))

            senses = [s for s in blocks[block_sig] if s != "__NOSENSE__" and isinstance(s, int)]
            senses.sort()
            if "__NOSENSE__" in blocks[block_sig]: senses.append("__NOSENSE__")

            ol_content = []
            for s in senses:
                rows_in_sense = blocks[block_sig][s]
                sense_children = {}

                seen_nonpos = set(); nonpos_badges = []
                for pr in rows_in_sense:
                    for i, tag in enumerate(pr["nonpos_tags"]):
                        if tag not in seen_nonpos:
                            seen_nonpos.add(tag)
                            color = pr["nonpos_colors"][i] if i < len(pr["nonpos_colors"]) else "#626273"
                            nonpos_badges.append(default_tag_span(tag, color))
                if nonpos_badges:
                    # Ensure order: tags before definitions/examples
                    pass

                ul_items = []
                def flush_ul():
                    nonlocal ul_items, sense_children
                    if ul_items:
                        sense_children.append({"tag": "ul", "content": ul_items})
                        ul_items = []

                # Switch to list for sense_children (typo above corrected)
                sense_children = []
                if nonpos_badges: sense_children.extend(nonpos_badges)

                for pr in rows_in_sense:
                    dn = pr["def_node"]
                    if dn.get("tag") == "li" and isinstance(dn.get("content"), str):
                        txt = prefix_if_narrower(pr["pos_tags"], list(block_sig), dn["content"])
                        ul_items.append({"tag": "li", "content": txt})
                    else:
                        ul_items.append(dn)

                    if pr["example"]:
                        flush_ul()
                        contoh, googlecontoh = pr["example"]
                        sense_children.append(example_block(contoh, googlecontoh, key))
                flush_ul()

                if s == "__NOSENSE__":
                    sense_li = {"tag": "li", "content": sense_children}
                else:
                    sense_li = {"tag": "li", "style": style_sense_li(make_circle_number(s)), "content": sense_children}
                ol_content.append(sense_li)

            if ol_content:
                block_content.append({"tag": "ol", "content": ol_content})
            top_ul["content"].append({"tag": "li", "content": block_content})

    # Bottom fields: each on its own line (li)
    for visible_label, field_key in BOTTOM_LABELS:
        raw = (pre.get(field_key) or "").strip()
        if not raw: continue
        tokens = [t.strip() for t in raw.split(" | ") if t.strip()]
        if not tokens: continue

        nodes = []
        for tok in tokens:
            node = build_anchor_or_text(tok, keys_lower, key_sense)
            if isinstance(node, list): nodes.extend(node)
            else: nodes.append(node)
        items = join_with_commas(nodes)
        top_ul["content"].append(bottom_line_item(visible_label, items))

    # Example-miss summary (non-spammy) will be printed in main() after writing
    top_children.append(top_ul)
    top_children.append(footer_block(key))

    structured = structured_root(top_children)
    return [key, pre.get("reading") or "", star, EMPTY_STR_AT_3, FIXED_VALUE_AT_4, [structured], int(pre.get("number_index") or 0), EMPTY_STR_AT_7]

# =========================
# Optional duplicate collapse (conservative)
# =========================
def collapse_identical_lines_in_structured(arr):
    try:
        sc = arr[5][0]
        children = sc["content"]
        if not children: return
        top_ul = children[0]
        for li in top_ul.get("content", []):
            for node in li.get("content", []):
                if isinstance(node, dict) and node.get("tag") == "ol":
                    for sense_li in node.get("content", []):
                        new_content = []
                        for ch in sense_li.get("content", []):
                            if isinstance(ch, dict) and ch.get("tag") == "ul":
                                new_ul = []
                                prev_text = None
                                for it in ch.get("content", []):
                                    if isinstance(it, dict) and it.get("tag") == "li" and isinstance(it.get("content"), str):
                                        cur = it["content"].strip()
                                        if prev_text is not None and cur == prev_text:
                                            warn("Collapsing duplicate identical definition line.")
                                            continue
                                        prev_text = cur
                                        new_ul.append(it)
                                    else:
                                        prev_text = None
                                        new_ul.append(it)
                                ch["content"] = new_ul
                                new_content.append(ch)
                            else:
                                new_content.append(ch)
                        sense_li["content"] = new_content
    except Exception as e:
        warn(f"collapse_identical_lines_in_structured encountered an issue: {e}")

# =========================
# Streaming writer
# =========================
class JsonArrayWriter:
    def __init__(self, folder: Path, base: str, limit_files: int, max_bytes: int):
        self.folder, self.base = folder, base
        self.limit, self.max_bytes = max(1, limit_files), max_bytes
        self.file_index, self.f, self.size, self.count = 1, None, 0, 0
        self.total_files_written = 0
        self._open_new_file()

    def _path(self) -> Path: return self.folder / f"{self.base}_{self.file_index}.json"

    def _open_new_file(self):
        if self.file_index > self.limit: self.f = None; return
        p = self._path()
        self.f = p.open("wb")
        self.f.write(b"["); self.size = 1; self.count = 0

    def _can_fit(self, entry_len: int) -> bool:
        extra = entry_len + (1 if self.count > 0 else 0)
        return self.size + extra + 1 <= self.max_bytes

    def write_entry(self, obj) -> bool:
        if self.f is None: return False
        b = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if not self._can_fit(len(b)):
            self.f.write(b"]"); self.f.close()
            info(f"Wrote {self._path().name} ({self.size + 1} bytes, {self.count} entries).")
            self.total_files_written += 1
            if self.file_index >= self.limit:
                info(f"Reached _TERM_BANK_GENERATE_LIMIT={self.limit}. Stopping.")
                return False
            self.file_index += 1
            self._open_new_file()
            if self.f is None: return False
            if not self._can_fit(len(b)):
                warn(f"Single entry exceeds {_MAX_JSON_BYTES} bytes; writing alone into {self._path().name}.")
        if self.count > 0: self.f.write(b","); self.size += 1
        self.f.write(b); self.size += len(b); self.count += 1
        return True

    def close(self):
        if self.f is not None:
            self.f.write(b"]"); self.f.close()
            info(f"Wrote {self._path().name} ({self.size + 1} bytes, {self.count} entries).")
            self.total_files_written += 1
            self.f = None

# =========================
# Main
# =========================
def main():
    workdir = Path(__file__).resolve().parent
    info(f"Reading CSVs from: {workdir}")

    pre_rows, ex_rows, link_rows, freq_rows, tag_rows, lite_rows = load_data(workdir)
    preprocess_by_key, keys_lower = index_preprocess(pre_rows)
    lite_by_key, key_sense       = index_lite(lite_rows)
    hyperlinks_by_row            = index_hyperlinks(link_rows)
    examples_by_row              = index_examples(ex_rows)
    freq_by_key                  = index_freq(freq_rows)
    tagmap                       = index_tags(tag_rows)

    keys = list(preprocess_by_key.keys())  # already somewhat ordered

    writer, processed = JsonArrayWriter(workdir, "term_bank", _TERM_BANK_GENERATE_LIMIT, _MAX_JSON_BYTES), 0
    for key in keys:
        pre = preprocess_by_key[key]
        rows = lite_by_key.get(key, [])

        arr = build_entry_array_for_key(
            key=key, pre=pre, lite_rows=rows,
            hyperlinks_by_row=hyperlinks_by_row, examples_by_row=examples_by_row,
            freq_by_key=freq_by_key, tagmap=tagmap, keys_lower=keys_lower, key_sense=key_sense
        )
        collapse_identical_lines_in_structured(arr)  # conservative

        if not writer.write_entry(arr): break
        processed += 1
        if processed % 1000 == 0: info(f"Processed {processed} keys...")
    writer.close()

    # Summarize suppressed example-miss warnings to avoid console spam
    if _MISS_SUPPRESSED:
        info(f"Suppressed {_MISS_SUPPRESSED} additional example highlight warnings (per-key and global caps).")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
