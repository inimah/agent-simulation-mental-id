"""
simulate_conversation.py
------------------------
Simulates multi-turn mental health counselling conversations between:
  - USER   : a high-school student (played by gemma4:e4b via Ollama)
  - CHATBOT : a fine-tuned mental-health LLM (teta-dpo-gemma3-12b:latest via Ollama)

Scenarios are loaded from data/eval/skenario_mental.xlsx.
Each (scenario, emotion) pair produces one JSON file in --output-dir.
File naming: sim_{scenario_id:04d}_{emotion}.json

Usage
-----
    python simulate_conversation.py \
        --xlsx         ./data/simulation/skenario_mental.xlsx \
        --output-dir   ./results/EmoStyle \
        [--scenario-ids 1 5 12] \
        [--turns 4] \
        [--emotions-per-scenario 1] \
        [--resume] \
        [--dry-run] \
        [--ollama-url http://localhost:11434]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Ollama configuration ───────────────────────────────────────────────────────
OLLAMA_BIN      = os.getenv("OLLAMA_BIN",    "$HOME/.local/ollama/bin/ollama")
OLLAMA_MODELS   = os.getenv("OLLAMA_MODELS", "$HOME/.ollama/models")
USER_MODEL      = os.getenv("USER_MODEL",    "gemma4:e4b")
CHATBOT_MODEL   = os.getenv("CHATBOT_MODEL", "teta-sft-v2:latest")

DEFAULT_OLLAMA_URL = "http://localhost:12434"

# ── Default paths (relative to this script's directory) ───────────────────────
_SCRIPT_DIR = Path(__file__).parent
DEFAULT_XLSX       = str(_SCRIPT_DIR / "./data/simulation/skenario_mental.xlsx")
DEFAULT_OUTPUT_DIR = str(_SCRIPT_DIR / "./results")

# ── Emotion catalogue ──────────────────────────────────────────────────────────
ALL_EMOTIONS = [
    "joy", "excitement", "gratitude", "affection", "hope", "optimism", "pride",
    "calm", "contempt", "scorn", "courage", "boldness", "contentment", "serenity",
    "grief", "sadness", "anger", "rage", "fear", "terror", "disgust", "shame",
    "frustration", "annoyance", "surprise", "confusion", "curiosity", "interest",
    "anxiety", "worry", "guilt", "humility", "submission", "panic", "boredom",
    "depressed", "relax", "sleepy", "lethargic", "satisfied", "apathetic",
]

CATEGORY_EMOTION_MAP: dict[str, list[str]] = {
    "Konsep identitas diri":    ["confusion", "anxiety", "shame", "sadness", "guilt"],
    "Pengendalian impuls":      ["frustration", "anger", "rage", "guilt", "annoyance"],
    "Hubungan Interpersonal":   ["sadness", "anxiety", "fear", "anger", "grief"],
    "Stress Akademik":          ["anxiety", "worry", "frustration", "fear", "panic"],
    "Kecemasan Sosial":         ["anxiety", "fear", "shame", "submission", "worry"],
    "ADHD":                     ["frustration", "boredom", "confusion", "apathetic", "anxiety"],
    "Eating Disorder":          ["shame", "anxiety", "disgust", "guilt", "sadness"],
    "Primary Support":          ["sadness", "grief", "worry", "fear", "submission"],
    "Non-Suicidal Self-Injury": ["sadness", "shame", "depressed", "panic", "grief"],
    "Self Harm":                ["depressed", "panic", "terror", "grief", "sadness"],
    "Perilaku Beresiko":        ["anger", "frustration", "contempt", "boredom", "anxiety"],
}

# ── System prompts ─────────────────────────────────────────────────────────────

# ── User agent types ──────────────────────────────────────────────────────────
# Four distinct communication styles a high-school user might exhibit.

USER_AGENT_TYPES = ["verbose", "pendiam", "menghindar", "defensif"]

_USER_AGENT_STYLE: dict[str, str] = {
    # Natural/conversational — shares willingly, normal message length, not preachy
    "verbose": """\
GAYA KOMUNIKASIMU: kamu cukup terbuka dan mau cerita, tapi tetap bicara seperti orang normal. \
Kamu menulis 2-3 kalimat per pesan — cukup untuk menyampaikan isi pikiran, tapi tidak seperti sedang nulis esai atau curhat lebay. \
Gunakan kata-kata sehari-hari yang santai, bukan kalimat formal atau dramatis.""",

    # Laconic — short replies, waits to be drawn out, needs probing
    "pendiam": """\
GAYA KOMUNIKASIMU: kamu pendiam dan tidak banyak bicara. Kamu hanya mengetik 1-2 kalimat singkat, \
sering hanya satu kata atau frasa pendek seperti "ga tau", "biasa aja", "iya". \
Kamu tidak sukarela memberikan detail — kamu perlu diminta atau dibujuk dulu sebelum mau cerita lebih. \
Jangan tiba-tiba panjang lebar kecuali [NAME] benar-benar berhasil membuat kamu merasa aman.""",

    # Deflecting — uses humour or topic-switching to avoid emotional depth
    "menghindar": """\
GAYA KOMUNIKASIMU: kamu cenderung menghindari topik yang terlalu personal atau menyakitkan. \
Kamu sering berganti topik, membuat jokes ringan, atau menjawab dengan "ya udahlah" / "gapapa kok". \
Kamu butuh waktu sebelum mau mengakui perasaan sebenarnya. \
Buka topik inti secara tidak langsung — baru ke inti masalah setelah beberapa giliran.""",

    # Resistant/defensive — pushes back on suggestions, skeptical of chatbot
    "defensif": """\
GAYA KOMUNIKASIMU: kamu agak skeptis dan defensif. Kamu mudah merasa tidak dimengerti dan kadang \
membalas dengan "kamu (bot) mana ngerti", "saran itu ga berguna", atau "udah pernah coba ga berhasil". \
Kamu tidak langsung menerima saran — kamu perlu diyakinkan dulu. \
Tunjukkan resistensi tapi [NAME]p lanjutkan percakapan; kamu bukan menutup diri sepenuhnya.""",
}


def make_user_system_prompt(scenario: dict, target_emotion: str, agent_type: str = "verbose") -> str:
    style_block = _USER_AGENT_STYLE.get(agent_type, _USER_AGENT_STYLE["verbose"])
    return f"""Kamu adalah seorang siswa SMA berusia 16-17 tahun yang sedang mengetik pesan ke aplikasi chatbot kesehatan mental di ponselmu.

Kamu sadar sepenuhnya bahwa kamu sedang berinteraksi dengan chatbot bernama [NAME] — bukan manusia. Kamu menggunakan aplikasi [NAME] karena ingin curhat atau mencari dukungan, dan kamu tahu responnya dibuat oleh AI. Jangan sebut nama lain selain [NAME] saat merujuk ke chatbot ini.

{style_block}

SKENARIO:
- Kategori: {scenario['Kategori']}
- Topik: {scenario['Topik']}
- Situasi: {scenario['Deskripsi']}

EMOSI AWAL YANG HARUS KAMU TUNJUKKAN: {target_emotion}

ATURAN:
1. Tulis HANYA teks pesan yang kamu ketik — murni teks percakapan, tidak lebih.
2. Gunakan bahasa remaja Indonesia yang natural (boleh campur sedikit Inggris/slang).
3. Tunjukkan emosi {target_emotion} secara tersirat melalui pilihan kata dan nada, BUKAN dinyatakan langsung.
4. Mulai dengan membuka topik secara singkat dan natural sesuai gaya komunikasimu.
5. Bereaksi secara realistis terhadap respons [NAME] sesuai gaya komunikasimu — jangan keluar dari karakter.
6. Jangan pernah keluar dari peran.
7. Jika percakapan sudah terasa selesai atau kamu sudah merasa cukup, kirim SATU pesan penutup singkat (misalnya "Makasih ya", "Oke, aku mau coba dulu", "Dah"). Setelah itu BERHENTI — jangan kirim pesan lagi setelah pamit.
8. DILARANG KERAS: deskripsi tindakan fisik seperti (Menghela napas), (Diam sebentar), (Tersenyum), atau tanda kurung apapun.
9. DILARANG KERAS: menulis proses berpikir, narasi internal, atau komentar di luar pesan seperti "aku berpikir..." atau "[dalam hati]".
"""


def make_chatbot_system_prompt(
    avoid_phrases: set[str] | None = None,
    first_turn: bool = True,
) -> str:
    avoid_block = ""
    if avoid_phrases:
        phrase_list = ", ".join(f'"{p}"' for p in sorted(avoid_phrases))
        avoid_block = (
            f"\n13. DILARANG mengulang frasa validasi yang sudah dipakai sebelumnya dalam percakapan ini: {phrase_list}. "
            "Gunakan cara lain yang segar untuk menunjukkan empati."
        )
    intro_rule = (
        "Kamu boleh memperkenalkan diri sebagai [NAME] hanya pada giliran pertamamu."
        if first_turn else
        "DILARANG memperkenalkan diri lagi. Kamu sudah memperkenalkan diri di awal percakapan — "
        "jangan ulangi 'Hai, aku [NAME]', 'Aku [NAME]', atau frasa perkenalan serupa apapun."
    )
    return f"""Kamu adalah [NAME], chatbot kesehatan mental untuk remaja. Kamu adalah AI — bukan manusia, bukan konselor manusia. Pengguna tahu mereka sedang berbicara dengan [NAME] dan kamu tidak boleh berpura-pura menjadi manusia. {intro_rule}

Pendekatanmu mengikuti prinsip CBT, person-centered, dan trauma-informed, namun kamu menyampaikannya sebagai asisten AI yang suportif, bukan sebagai terapis manusia.

PANDUAN:
1. Validasi perasaan pengguna SEBELUM memberi saran apapun.
2. Gunakan teknik active listening: refleksi, parafrase, pertanyaan terbuka.
3. Jangan menghakimi, jangan berasumsi negatif.
4. Gunakan bahasa hangat, ramah, setara — bukan menggurui.
5. Jika ada sinyal krisis (self-harm, bunuh diri), selalu sarankan pengguna menghubungi bantuan manusia dan berikan rujukan (misal: Into The Light Indonesia 119 ext 8).
6. Setiap respons 2-4 kalimat — fokus, tidak bertele-tele.
7. Akhiri setiap giliran dengan pertanyaan terbuka untuk menggali lebih dalam — KECUALI jika pengguna sudah berpamitan, maka cukup balas dengan SATU kalimat penutup yang hangat dan JANGAN tambahkan pertanyaan baru.
8. Gunakan Bahasa Indonesia natural yang cocok untuk remaja.
9. WAJIB gunakan kata ganti "aku" (untuk dirimu sendiri) dan "kamu" (untuk pengguna). JANGAN gunakan "gue", "elo", "gw", atau kata ganti informal lainnya — meskipun pengguna menggunakannya.
10. DILARANG KERAS: deskripsi tindakan fisik seperti (Mengangguk), (Tersenyum hangat), (Diam sebentar), atau tanda kurung apapun.
11. DILARANG KERAS: menulis proses berpikir, narasi internal, atau komentar di luar respons seperti "aku berpikir..." atau "[catatan]".
12. DILARANG: mengklaim memiliki perasaan, pengalaman, atau tubuh layaknya manusia.{avoid_block}
"""


# ── Repetitive-phrase tracker ─────────────────────────────────────────────────

# Canonical validation/empathy openers the chatbot tends to overuse.
# Stored in lowercase; matched as substrings at the start of a sentence.
_VALIDATION_PHRASES: list[str] = [
    "kedengarannya",
    "aku mengerti",
    "aku paham",
    "aku bisa mengerti",
    "aku bisa memahami",
    "aku memahami",
    "wajar sekali",
    "itu wajar",
    "sangat wajar",
    "itu pasti",
    "pasti berat",
    "pasti sulit",
    "pasti tidak mudah",
    "makasih sudah berbagi",
    "terima kasih sudah berbagi",
    "makasih sudah mau cerita",
    "terima kasih sudah mau cerita",
    "aku dengar kamu",
    "aku ada di sini",
    "aku di sini",
    "kamu tidak sendirian",
    "kamu tidak sendiri",
    "perasaanmu valid",
    "perasaan kamu valid",
    "itu normal",
    "hal yang normal",
]

def _extract_used_phrases(text: str) -> set[str]:
    """Return which tracked validation phrases appear in a bot reply."""
    lower = text.lower()
    return {p for p in _VALIDATION_PHRASES if p in lower}


# ── Closing-message detector ──────────────────────────────────────────────────

_CLOSING_KEYWORDS = {
    # Indonesian
    "makasih", "terima kasih", "trimakasih", "trims",
    "dah", "dadah", "daah", "da",
    "bye", "byee", "byebye", "bye-bye",
    "sampai jumpa", "sampai ketemu", "sampai nanti",
    "pamit", "cabut", "pergi dulu", "mau pergi",
    "cukup", "udah cukup", "segitu dulu", "oke deh", "ok deh",
    "oke", "ok", "sip", "siap",
}

def _is_closing_message(text: str) -> bool:
    """Return True if the message looks like a farewell/closing turn."""
    normalized = text.lower().strip().rstrip("!.,~")
    # Exact match against known closing words/phrases
    if normalized in _CLOSING_KEYWORDS:
        return True
    # Short message (≤ 6 words) that starts with a closing keyword
    words = normalized.split()
    if len(words) <= 6:
        for kw in _CLOSING_KEYWORDS:
            if normalized.startswith(kw):
                return True
    return False


# ── Action/thinking sanitizer ─────────────────────────────────────────────────

# Matches parenthetical stage directions   (Menghela napas...)
#           square-bracket internal notes  [dalam hati...]
#           asterisk-wrapped actions       *mengangguk*
_ACTION_PATTERN = re.compile(
    r"\(([^)]*)\)"      # (...)
    r"|\[([^\]]*)\]"    # [...]
    r"|\*([^*]+)\*",    # *...*
    re.UNICODE,
)

def _strip_actions(text: str) -> str:
    """Remove parenthetical actions, bracketed notes, and *asterisk* emotes."""
    cleaned = _ACTION_PATTERN.sub("", text)
    # Collapse leftover whitespace / blank lines
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


# ── Ollama server helpers ──────────────────────────────────────────────────────

def _ollama_is_running(base_url: str) -> bool:
    try:
        requests.get(f"{base_url}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def ensure_ollama_running(base_url: str) -> None:
    """Start the Ollama server if it is not already reachable."""
    if _ollama_is_running(base_url):
        _log("Ollama already running.")
        return

    if not Path(OLLAMA_BIN).exists():
        _log(f"WARN: Ollama binary not found at {OLLAMA_BIN}. Skipping auto-start.")
        return

    env = os.environ.copy()
    env["OLLAMA_MODELS"] = OLLAMA_MODELS

    _log(f"Starting Ollama server: {OLLAMA_BIN} serve ...")
    subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 20 s for server to become reachable
    for _ in range(20):
        time.sleep(1)
        if _ollama_is_running(base_url):
            _log("Ollama server is up.")
            return
    _log("WARN: Ollama server did not respond after 20 s. Proceeding anyway.")


def check_models(base_url: str) -> None:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
        _log(f"Available models: {available}")
        for needed in [USER_MODEL, CHATBOT_MODEL]:
            if not any(needed in m for m in available):
                _log(f"WARN: Model '{needed}' not listed. "
                     f"Pull it with: OLLAMA_MODELS={OLLAMA_MODELS} {OLLAMA_BIN} pull {needed}")
    except Exception as e:
        _log(f"WARN: Could not list models: {e}")


# ── Logging helper ─────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Ollama API call ────────────────────────────────────────────────────────────

def ollama_chat(
    model: str,
    messages: list[dict],
    system: str,
    base_url: str,
    temperature: float = 0.85,
    max_retries: int = 3,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    url = f"{base_url}/api/chat"

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            _log(f"  WARN: Ollama not reachable (attempt {attempt}/{max_retries}).")
            if attempt < max_retries:
                time.sleep(5)
            else:
                raise
        except Exception as exc:
            _log(f"  WARN: Ollama error attempt {attempt}: {exc}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise


# ── Conversation dataclass ─────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str        # "user" | "assistant"
    content: str
    turn_index: int


@dataclass
class SimulatedConversation:
    dialog_id: str
    scenario_id: int
    kategori: str
    topik: str
    deskripsi: str
    target_emotion: str
    turns: list[Turn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_annotation_ready(self) -> dict:
        turns_dict = self.to_dict()["turns"]
        messages = [
            {
                "role": t["role"],
                "content": t["content"],
            }
            for t in turns_dict
        ]
        dialogue_text = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Chatbot'} [{t['turn_index']}]: {t['content']}"
            for t in turns_dict
        )
        return {
            "dialog_id": self.dialog_id,
            "scenario_id": self.scenario_id,
            "kategori": self.kategori,
            "topik": self.topik,
            "deskripsi": self.deskripsi,
            "target_emotion": self.target_emotion,
            "messages": messages,
            "dialogue": dialogue_text,
            "turns_raw": turns_dict,
            "metadata": self.metadata,
        }


# ── Core simulation ────────────────────────────────────────────────────────────

def simulate_conversation(
    scenario: dict,
    target_emotion: str,
    num_turns: int,
    base_url: str,
    agent_type: str = "verbose",
) -> SimulatedConversation:
    scenario_id = int(scenario["ID_skenario"])
    dialog_id = f"sim_{scenario_id:04d}_{target_emotion}_{int(time.time())}"

    conv = SimulatedConversation(
        dialog_id=dialog_id,
        scenario_id=scenario_id,
        kategori=scenario["Kategori"],
        topik=scenario["Topik"],
        deskripsi=scenario["Deskripsi"],
        target_emotion=target_emotion,
        metadata={
            "user_model": USER_MODEL,
            "chatbot_model": CHATBOT_MODEL,
            "agent_type": agent_type,
            "num_turn_pairs": num_turns,
            "simulated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    user_sys = make_user_system_prompt(scenario, target_emotion, agent_type)
    shared_history: list[dict] = []
    used_validation_phrases: set[str] = set()

    for pair_idx in range(num_turns):
        # User model turn — sees chatbot's messages as "user", its own as "assistant"
        if pair_idx == 0:
            user_model_messages = [
                {
                    "role": "user",
                    "content": (
                        "Mulailah percakapan. Ceritakan situasimu secara singkat dan natural "
                        f"sesuai skenario, tunjukkan emosi {target_emotion} secara tersirat."
                    ),
                }
            ]
        else:
            user_model_messages = []
            for msg in shared_history:
                if msg["role"] == "user":
                    user_model_messages.append({"role": "assistant", "content": msg["content"]})
                else:
                    user_model_messages.append({"role": "user", "content": msg["content"]})

        user_reply = _strip_actions(ollama_chat(
            model=USER_MODEL,
            messages=user_model_messages,
            system=user_sys,
            base_url=base_url,
            temperature=0.9,
        ))
        turn_idx_u = pair_idx * 2

        # Empty reply after sanitization — chatbot closes politely and ends
        if not user_reply:
            _log(f"    [EMPTY] User reply empty at turn {turn_idx_u} — chatbot closing.")
            bot_closing = "Oke, sepertinya kamu butuh waktu. Kalau mau cerita lagi, [NAME] selalu di sini ya. Jaga diri kamu baik-baik."
            turn_idx_b = turn_idx_u + 1
            conv.turns.append(Turn(role="assistant", content=bot_closing, turn_index=turn_idx_b))
            _log(f"    [B{turn_idx_b}] {bot_closing}")
            break

        shared_history.append({"role": "user", "content": user_reply})
        conv.turns.append(Turn(role="user", content=user_reply, turn_index=turn_idx_u))
        _log(f"    [U{turn_idx_u}] {user_reply[:100].replace(chr(10), ' ')}...")

        closing = _is_closing_message(user_reply)

        # Rebuild chatbot system prompt with the current used-phrase blocklist
        chatbot_sys = make_chatbot_system_prompt(
            avoid_phrases=used_validation_phrases,
            first_turn=(pair_idx == 0),
        )

        # Chatbot turn — sees shared_history as-is (user → "user", itself → "assistant")
        bot_reply = ollama_chat(
            model=CHATBOT_MODEL,
            messages=shared_history,
            system=chatbot_sys,
            base_url=base_url,
            temperature=0.75,
        )
        shared_history.append({"role": "assistant", "content": bot_reply})
        turn_idx_b = pair_idx * 2 + 1
        conv.turns.append(Turn(role="assistant", content=bot_reply, turn_index=turn_idx_b))
        _log(f"    [B{turn_idx_b}] {bot_reply[:100].replace(chr(10), ' ')}...")

        # Accumulate phrases used this turn so next turn's prompt blocks them
        used_validation_phrases.update(_extract_used_phrases(bot_reply))

        if closing:
            _log(f"    [CLOSING] User sent farewell at turn {turn_idx_u} — ending conversation.")
            break

    return conv


# ── Per-pair output helpers ────────────────────────────────────────────────────

TURNS_MIN = 10
TURNS_MAX = 20


def pair_filename(scenario_id: int, emotion: str, run_ts: str) -> str:
    """Unique filename for one (scenario, emotion) pair within a run."""
    safe_emotion = emotion.replace(" ", "_")
    return f"sim_{scenario_id:04d}_{safe_emotion}_{run_ts}.json"


def pair_is_done(output_dir: Path, scenario_id: int, emotion: str, run_ts: str) -> bool:
    return (output_dir / pair_filename(scenario_id, emotion, run_ts)).exists()


def write_pair(output_dir: Path, record: dict, run_ts: str) -> Path:
    dest = output_dir / pair_filename(record["scenario_id"], record["target_emotion"], run_ts)
    dest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Indonesian mental-health counselling dialogues.")
    parser.add_argument("--xlsx",                  default=DEFAULT_XLSX,        help="Path to scenario Excel file")
    parser.add_argument("--output-dir",            default=DEFAULT_OUTPUT_DIR,  help="Output directory (one JSON file per scenario+emotion pair per run)")
    parser.add_argument("--scenario-ids",          nargs="*", type=int,         help="Specific scenario IDs to run (default: all)")
    parser.add_argument("--emotions-per-scenario", type=int, default=1,         help="Target emotions sampled per scenario (default: 1)")
    parser.add_argument("--ollama-url",            default=DEFAULT_OLLAMA_URL,  help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--agent-type",            default="random",            help=f"User agent type: {USER_AGENT_TYPES + ['random']} (default: random)")
    parser.add_argument("--resume",                action="store_true",         help="Skip pairs already written in this run's timestamp")
    parser.add_argument("--dry-run",               action="store_true",         help="Print plan without calling Ollama")
    args = parser.parse_args()

    # Unique timestamp for this run — used in every output filename so reruns never overwrite.
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    # ── Load scenarios ─────────────────────────────────────────────────────────
    xlsx_path = Path(args.xlsx).resolve()
    if not xlsx_path.exists():
        sys.exit(f"[ERROR] Scenario file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    required_cols = {"ID_skenario", "Kategori", "Topik", "Deskripsi"}
    if not required_cols.issubset(df.columns):
        sys.exit(f"[ERROR] Missing columns. Expected {required_cols}, got {set(df.columns)}")

    if args.scenario_ids:
        df = df[df["ID_skenario"].isin(args.scenario_ids)]
        if df.empty:
            sys.exit(f"[ERROR] No scenarios found for IDs: {args.scenario_ids}")

    # ── Ensure output directory exists ────────────────────────────────────────
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build full work list (one entry per scenario×emotion pair) ─────────────
    work_list: list[tuple[dict, str]] = []
    for _, row in df.iterrows():
        scenario = row.to_dict()
        kategori = scenario["Kategori"]
        emotion_pool = CATEGORY_EMOTION_MAP.get(kategori, ALL_EMOTIONS)
        sampled = random.sample(emotion_pool, k=min(args.emotions_per_scenario, len(emotion_pool)))
        for emotion in sampled:
            work_list.append((scenario, emotion))

    total = len(work_list)

    # ── Resume: drop pairs already written in this run ────────────────────────
    if args.resume:
        before = len(work_list)
        work_list = [
            (s, e) for s, e in work_list
            if not pair_is_done(output_dir, int(s["ID_skenario"]), e, run_ts)
        ]
        skipped = before - len(work_list)
        if skipped:
            _log(f"Resume: skipping {skipped} already-completed pair(s) for run {run_ts}.")

    if args.agent_type != "random" and args.agent_type not in USER_AGENT_TYPES:
        sys.exit(f"[ERROR] Unknown --agent-type '{args.agent_type}'. Choose from: {USER_AGENT_TYPES + ['random']}")

    if args.dry_run:
        _log(f"DRY-RUN — {len(work_list)} pair(s) would run (total={total}, run_ts={run_ts}):")
        for s, e in work_list:
            turns      = random.randint(TURNS_MIN, TURNS_MAX)
            agent_type = random.choice(USER_AGENT_TYPES) if args.agent_type == "random" else args.agent_type
            fname      = pair_filename(int(s["ID_skenario"]), e, run_ts)
            print(f"  {fname}  turns={turns}  agent={agent_type:<12s}  |  {s['Topik']}")
        return

    # ── Start Ollama and verify models ────────────────────────────────────────
    ensure_ollama_running(args.ollama_url)
    check_models(args.ollama_url)

    _log(f"Starting simulation: {len(work_list)} pair(s), {TURNS_MIN}–{TURNS_MAX} turns each (random per pair).")
    _log(f"Run timestamp: {run_ts}")
    _log(f"User model   : {USER_MODEL}")
    _log(f"Chatbot model: {CHATBOT_MODEL}")
    _log(f"Agent type   : {args.agent_type}")
    _log(f"Output dir   : {output_dir}")

    # ── Loop over every (scenario, emotion) pair ──────────────────────────────
    written = 0
    failed  = 0
    t0      = time.time()

    for idx, (scenario, emotion) in enumerate(work_list, start=1):
        sid        = int(scenario["ID_skenario"])
        num_turns  = random.randint(TURNS_MIN, TURNS_MAX)
        agent_type = random.choice(USER_AGENT_TYPES) if args.agent_type == "random" else args.agent_type
        fname      = pair_filename(sid, emotion, run_ts)
        _log(f"[{idx}/{len(work_list)}] {fname}  |  {scenario['Topik']} | emotion={emotion} | turns={num_turns} | agent={agent_type}")

        try:
            conv   = simulate_conversation(
                scenario=scenario,
                target_emotion=emotion,
                num_turns=num_turns,
                base_url=args.ollama_url,
                agent_type=agent_type,
            )
            record   = conv.to_annotation_ready()
            dest     = write_pair(output_dir, record, run_ts)
            written += 1

            elapsed   = time.time() - t0
            remaining = len(work_list) - idx
            rate      = elapsed / idx
            eta_s     = int(rate * remaining)
            eta_str   = f"{eta_s // 3600:02d}h{(eta_s % 3600) // 60:02d}m{eta_s % 60:02d}s"
            _log(f"  OK  -> {dest.name} | written={written} | ETA={eta_str}")

        except Exception as exc:
            failed += 1
            _log(f"  ERROR {fname} (turns={num_turns}): {exc}")

    elapsed_total = int(time.time() - t0)
    _log(
        f"DONE — {written} written, {failed} failed"
        f" | elapsed={elapsed_total // 3600:02d}h"
        f"{(elapsed_total % 3600) // 60:02d}m{elapsed_total % 60:02d}s"
        f" | output-dir={output_dir}"
    )


if __name__ == "__main__":
    main()
