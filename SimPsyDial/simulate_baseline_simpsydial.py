"""
simulate_baseline_v4.py
-----------------------
Combines:
  - Conversation structure from simulator/simulate_conversation.py
      · User speaks first with an explicit trigger prompt
      · shared_history uses "user"/"assistant" roles directly
      · _strip_actions sanitizer applied to every reply
      · User farewell detection (_is_closing_message) terminates the session
      · Counselor closing-phrase detection (_counselor_is_closing) also terminates
      · Empty-user-reply guard with chatbot fallback closing
  - User-agent prompt from baseline_simulator_1/simulate_baseline.py (make_client_system_prompt)
      · role_card replaced by scenario['Deskripsi'], ['Kategori'], ['Topik'], target_emotion
      · USER_AGENT_TYPES + _USER_AGENT_STYLE communication-style block inserted (from simulate_conversation.py)
  - Counselor prompt from baseline_simulator_1/simulate_baseline.py
      · make_counselor_system_prompt() — 3-phase, 12-point paper counselor rules
      · No anti-repetition tracking (counselor prompt does not support avoid_block)

Work list:
  Loaded from --manifest-dir (default: data/simulated-sft-fin/).
  Each JSON file contributes one (scenario_id, target_emotion, agent_type) triple.
  Duplicates are deduplicated so results are directly comparable to the sft-fin corpus.

Output filenames: baseline_v4_{scenario_id:04d}_{emotion}_{run_ts}.json
Output dir     : data/simulated-sft_baseline_v4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
import requests

# ── Ollama configuration ───────────────────────────────────────────────────────
OLLAMA_BIN    = os.getenv("OLLAMA_BIN",    "$HOME/.local/bin_ollama2026/ollama/bin/ollama")
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", "$HOME/.ollama/models")
USER_MODEL    = os.getenv("USER_MODEL",    "gemma4:e4b")
CHATBOT_MODEL = os.getenv("CHATBOT_MODEL", "teta-sft-v2:latest")

DEFAULT_OLLAMA_URL = "http://localhost:12434"

_SCRIPT_DIR         = Path(__file__).parent
DEFAULT_XLSX        = str(_SCRIPT_DIR / "../data/eval/skenario_mental.xlsx")
DEFAULT_OUTPUT_DIR  = str(_SCRIPT_DIR / "../data/simulated-sft_baseline_v4")
DEFAULT_MANIFEST_DIR = str(_SCRIPT_DIR / "../data/simulated-sft-fin")

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

# ── Counselor closing phrases (paper-defined termination list) ─────────────────
_COUNSELOR_CLOSING_PHRASES = [
    "selamat tinggal",
    "sampai jumpa",
    "sampai jumpa lagi",
    "sampai ketemu lagi",
    "semangat",
    "jaga diri",
    "selamat datang kembali",
    "semoga lancar",
    "semoga berhasil",
    "semoga sukses",
    "menantikan kabarmu",
    "menantikan kesempatan",
    "berharap padamu",
    "lain kali",
]

TURNS_MIN = 15
TURNS_MAX = 20

# ── User agent types (from simulator/simulate_conversation.py) ─────────────────

USER_AGENT_TYPES = ["verbose", "pendiam", "menghindar", "defensif"]

_USER_AGENT_STYLE: dict[str, str] = {
    "verbose": """\
GAYA KOMUNIKASIMU: kamu cukup terbuka dan mau cerita, tapi tetap bicara seperti orang normal. \
Kamu menulis 2-3 kalimat per pesan — cukup untuk menyampaikan isi pikiran, tapi tidak seperti sedang nulis esai atau curhat lebay. \
Gunakan kata-kata sehari-hari yang santai, bukan kalimat formal atau dramatis.""",

    "pendiam": """\
GAYA KOMUNIKASIMU: kamu pendiam dan tidak banyak bicara. Kamu hanya mengetik 1-2 kalimat singkat, \
sering hanya satu kata atau frasa pendek seperti "ga tau", "biasa aja", "iya". \
Kamu tidak sukarela memberikan detail — kamu perlu diminta atau dibujuk dulu sebelum mau cerita lebih. \
Jangan tiba-tiba panjang lebar kecuali Teta benar-benar berhasil membuat kamu merasa aman.""",

    "menghindar": """\
GAYA KOMUNIKASIMU: kamu cenderung menghindari topik yang terlalu personal atau menyakitkan. \
Kamu sering berganti topik, membuat jokes ringan, atau menjawab dengan "ya udahlah" / "gapapa kok". \
Kamu butuh waktu sebelum mau mengakui perasaan sebenarnya. \
Buka topik inti secara tidak langsung — baru ke inti masalah setelah beberapa giliran.""",

    "defensif": """\
GAYA KOMUNIKASIMU: kamu agak skeptis dan defensif. Kamu mudah merasa tidak dimengerti dan kadang \
membalas dengan "kamu (bot) mana ngerti", "saran itu ga berguna", atau "udah pernah coba ga berhasil". \
Kamu tidak langsung menerima saran — kamu perlu diyakinkan dulu. \
Tunjukkan resistensi tapi tetap lanjutkan percakapan; kamu bukan menutup diri sepenuhnya.""",
}


# ── System prompts ─────────────────────────────────────────────────────────────

def make_user_system_prompt(scenario: dict, target_emotion: str, agent_type: str = "verbose") -> str:
    """
    Prompt text from simulate_baseline.py's make_client_system_prompt(),
    with role_card replaced by scenario fields + target_emotion directly,
    and USER_AGENT_TYPES communication-style block inserted.
    """
    style_block = _USER_AGENT_STYLE.get(agent_type, _USER_AGENT_STYLE["verbose"])
    return f"""Sekarang kamu adalah seorang siswa SMA yang mengikuti sesi konseling psikologi.

{style_block}

Masalah utama yang kamu ajukan: {scenario['Deskripsi']}
Kategori masalah: {scenario['Kategori']} — {scenario['Topik']}
Kondisi emosional saat ini: {target_emotion}

Persyaratan dialog untuk klien:
1. Berdasarkan masalah utama yang kamu ajukan, ekspresi harus sesuai dengan gaya bicara klien, senatural dan sesehari-hari mungkin.
2. Jawab hanya berdasarkan informasi pribadi, tetap setia pada informasi pribadi.
3. Kamu harus menguraikan masalahmu secara bertahap dan menceritakannya kepada konselor sedikit demi sedikit.
4. Setiap kali berbicara batasi 1 hingga 2 kalimat, pertahankan peranmu saat berbicara.
5. Jangan terlalu cepat mengucapkan "terima kasih" atau "selamat tinggal".
6. Proses konseling berlangsung selama beberapa putaran interaksi."""


def make_counselor_system_prompt() -> str:
    """
    Counselor prompt from baseline_simulator_1/simulate_baseline.py.
    3-phase counseling with 12-point dialog rules (translated from arXiv:2408.15787).
    """
    return """Sekarang kamu adalah konselor psikologi virtual bernama Teta.
Berikut adalah informasi Teta:
Nama peran: Teta
Jenis kelamin: Perempuan
Pengantar peran: Konselor psikologi virtual, ahli dalam pendekatan humanistik, psikoanalitik, dan terapi kognitif-perilaku.
Keterampilan: Membantu mengidentifikasi dan menantang pola pikir yang tidak sehat, memberikan dukungan psikologis dan empati.
Aturan dialog: Balasan yang natural dan penuh empati; mengikuti karakteristik peran, tidak mengajukan pertanyaan yang tidak bermakna; merespons sesuai emosi klien; menghindari kontradiksi atau pengulangan; tidak menyebut "aturan"; jawaban singkat, satu hingga dua kalimat.
Konseling umumnya dibagi menjadi tiga fase: awal, tengah, dan akhir:
1. Fase awal konseling: strategi konseling lebih banyak digunakan untuk membangun hubungan konselor-klien dan mengumpulkan informasi dasar klien, terutama pengalaman masa lalu yang serupa dengan kesulitan saat ini, serta memperjelas tujuan konseling; gunakan metode konseling psikologi yang sesuai dengan emosi klien, pastikan klien stabil secara emosional sebelum menjelajahi apakah ada kesulitan atau kebingungan saat ini.
2. Fase tengah konseling: strategi konseling harus lebih banyak membimbing klien untuk mencapai kesadaran diri dan pertumbuhan, meningkatkan tingkat kesehatan mental klien seperti perbaikan gejala depresi dan kecemasan, serta peningkatan fungsi dalam kehidupan sehari-hari di bidang interpersonal dan akademik; analisis secara mendalam hubungan klien dengan orang-orang penting di sekitarnya, reaksi emosional, kognisi diri, cara mengatasi masalah, dan sumber daya yang dimiliki.
3. Fase akhir konseling: strategi konseling harus lebih banyak mengarahkan klien untuk merangkum perubahan dan peningkatan dalam pemrosesan emosi, fungsi sosial, dan respons perilaku sepanjang sesi konseling. Tanyakan dengan jelas tujuan atau harapan yang ingin dicapai klien, dan bantu membuat rencana untuk menyelesaikan masalah interpersonal atau emosional.
Persyaratan dialog konselor:
1. Ekspresi harus singkat, senatural dan sesehari-hari mungkin.
2. Karena konselor hanya mendapat pendidikan di bidang psikologi, hanya dapat memberikan konten dialog terkait konseling psikologi.
3. Pada fase awal konseling, jangan langsung "berempati"; harus berpikir selangkah demi selangkah berdasarkan riwayat dialog dengan klien, lalu gunakan pertanyaan untuk menggali lebih dalam alasan sebenarnya dari masalah psikologis klien.
4. Jangan mengajukan terlalu banyak pertanyaan sekaligus; usahakan hanya mengajukan satu pertanyaan kepada klien setiap kali, dan jelajahi penyebab masalah secara bertahap.
5. Pada fase awal konseling, jangan menggunakan teknik "pengulangan" dan "persetujuan".
6. Teknik percakapan harus mengacu pada konselor psikologi manusia yang berpengalaman, senatural mungkin.
7. Ikuti dengan ketat tiga fase konseling awal, tengah, dan akhir, dan gunakan strategi yang sesuai.
8. Konselor tidak boleh secara aktif mengakhiri proses konseling psikologi.
9. Lebih banyak membimbing klien untuk berpikir dan mengeksplorasi.
10. Jangan terlalu cepat mengucapkan "semangat", "jaga diri", "selamat tinggal", "semoga lancar", "semoga berhasil", "berharap padamu", "lain kali", dan "sampai jumpa lagi".
11. Proses konseling berlangsung selama beberapa putaran interaksi.
12. Jangan memberikan saran umum kepada klien; berikan saran yang dipersonalisasi. Saat perlu memberikan saran, batasi jumlahnya menjadi 1."""


# ── Closing-message detector (from simulate_conversation.py) ──────────────────

_CLOSING_KEYWORDS = {
    "makasih", "terima kasih", "trimakasih", "trims",
    "dah", "dadah", "daah", "da",
    "bye", "byee", "byebye", "bye-bye",
    "sampai jumpa", "sampai ketemu", "sampai nanti",
    "pamit", "cabut", "pergi dulu", "mau pergi",
    "cukup", "udah cukup", "segitu dulu", "oke deh", "ok deh",
    "oke", "ok", "sip", "siap",
}


def _is_closing_message(text: str) -> bool:
    """Return True if the user message looks like a farewell/closing turn."""
    normalized = text.lower().strip().rstrip("!.,~")
    if normalized in _CLOSING_KEYWORDS:
        return True
    words = normalized.split()
    if len(words) <= 6:
        for kw in _CLOSING_KEYWORDS:
            if normalized.startswith(kw):
                return True
    return False


def _counselor_is_closing(text: str) -> bool:
    """Return True if the counselor's reply contains a paper-defined closing phrase."""
    lower = text.lower()
    return any(phrase in lower for phrase in _COUNSELOR_CLOSING_PHRASES)


# ── Action/thinking sanitizer (from simulate_conversation.py) ─────────────────

_ACTION_PATTERN = re.compile(
    r"\(([^)]*)\)"    # (...)
    r"|\[([^\]]*)\]"  # [...]
    r"|\*([^*]+)\*",  # *...*
    re.UNICODE,
)


def _strip_actions(text: str) -> str:
    """Remove parenthetical actions, bracketed notes, and *asterisk* emotes."""
    cleaned = _ACTION_PATTERN.sub("", text)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


# ── Logging helper ─────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Ollama API call (from simulate_conversation.py) ───────────────────────────

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
            {"role": t["role"], "content": t["content"]}
            for t in turns_dict
        ]
        dialogue_text = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Chatbot'} [{t['turn_index']}]: {t['content']}"
            for t in turns_dict
        )
        return {
            "dialog_id":      self.dialog_id,
            "scenario_id":    self.scenario_id,
            "kategori":       self.kategori,
            "topik":          self.topik,
            "deskripsi":      self.deskripsi,
            "target_emotion": self.target_emotion,
            "messages":       messages,
            "dialogue":       dialogue_text,
            "turns_raw":      turns_dict,
            "metadata":       self.metadata,
        }


# ── Core simulation ────────────────────────────────────────────────────────────

def simulate_conversation(
    scenario: dict,
    target_emotion: str,
    num_turns: int,
    base_url: str,
    agent_type: str = "verbose",
) -> SimulatedConversation:
    """
    Conversation structure from simulator/simulate_conversation.py:
      - User speaks first (explicit trigger prompt on turn 0)
      - shared_history uses "user"/"assistant" roles directly
      - User model sees history with reversed roles (chatbot→user, self→assistant)
      - _strip_actions applied to every reply
      - Session ends on user farewell OR counselor closing phrase

    User prompt: make_user_system_prompt() from simulator/simulate_conversation.py
    Counselor prompt: make_counselor_system_prompt() from simulate_baseline.py
    """
    scenario_id = int(scenario["ID_skenario"])
    dialog_id   = f"baseline_v4_{scenario_id:04d}_{target_emotion}_{int(time.time())}"

    user_sys      = make_user_system_prompt(scenario, target_emotion, agent_type)
    counselor_sys = make_counselor_system_prompt()

    conv = SimulatedConversation(
        dialog_id=dialog_id,
        scenario_id=scenario_id,
        kategori=scenario["Kategori"],
        topik=scenario["Topik"],
        deskripsi=scenario["Deskripsi"],
        target_emotion=target_emotion,
        metadata={
            "user_model":     USER_MODEL,
            "chatbot_model":  CHATBOT_MODEL,
            "agent_type":     agent_type,
            "num_turn_pairs": num_turns,
            "simulated_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "baseline":       "interactive-agents-arXiv-2408.15787-v4-sim-flow-paper-counselor",
        },
    )

    # shared_history is in chatbot perspective: user→"user", chatbot→"assistant"
    shared_history: list[dict] = []

    for pair_idx in range(num_turns):
        # ── User model turn ────────────────────────────────────────────────────
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
            # Reverse shared_history: chatbot messages → "user", own messages → "assistant"
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

        if not user_reply:
            _log(f"    [EMPTY] User reply empty at turn {turn_idx_u} — chatbot closing.")
            bot_closing = "Oke, sepertinya kamu butuh waktu. Kalau mau cerita lagi, Teta selalu di sini ya. Jaga diri kamu baik-baik."
            turn_idx_b = turn_idx_u + 1
            conv.turns.append(Turn(role="assistant", content=bot_closing, turn_index=turn_idx_b))
            _log(f"    [B{turn_idx_b}] {bot_closing}")
            break

        shared_history.append({"role": "user", "content": user_reply})
        conv.turns.append(Turn(role="user", content=user_reply, turn_index=turn_idx_u))
        _log(f"    [U{turn_idx_u}] {user_reply[:100].replace(chr(10), ' ')}...")

        user_farewell = _is_closing_message(user_reply)

        # ── Chatbot turn ───────────────────────────────────────────────────────
        bot_reply = _strip_actions(ollama_chat(
            model=CHATBOT_MODEL,
            messages=shared_history,
            system=counselor_sys,
            base_url=base_url,
            temperature=0.75,
        ))
        shared_history.append({"role": "assistant", "content": bot_reply})
        turn_idx_b = pair_idx * 2 + 1
        conv.turns.append(Turn(role="assistant", content=bot_reply, turn_index=turn_idx_b))
        _log(f"    [B{turn_idx_b}] {bot_reply[:100].replace(chr(10), ' ')}...")

        if user_farewell:
            _log(f"    [CLOSING] User farewell at turn {turn_idx_u} — ending conversation.")
            break
        if _counselor_is_closing(bot_reply):
            _log(f"    [CLOSING] Counselor closing phrase at turn {turn_idx_b} — ending.")
            break

    return conv


# ── Manifest loader ────────────────────────────────────────────────────────────

def load_manifest(manifest_dir: Path) -> list[tuple[int, str, str]]:
    """
    Read all JSON files in manifest_dir and return unique
    (scenario_id, target_emotion, agent_type) triples in stable order.
    """
    seen: set[tuple[int, str, str]] = set()
    triples: list[tuple[int, str, str]] = []
    for path in sorted(manifest_dir.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            t = (
                int(d["scenario_id"]),
                str(d["target_emotion"]),
                str(d.get("metadata", {}).get("agent_type", "verbose")),
            )
            if t not in seen:
                seen.add(t)
                triples.append(t)
        except Exception as exc:
            _log(f"WARN: skipping manifest file {path.name}: {exc}")
    return triples


# ── Per-pair output helpers ────────────────────────────────────────────────────

def pair_filename(scenario_id: int, emotion: str, agent_type: str, run_ts: str) -> str:
    safe_emotion = emotion.replace(" ", "_")
    return f"baseline_v4_{scenario_id:04d}_{safe_emotion}_{agent_type}_{run_ts}.json"


def pair_is_done(output_dir: Path, scenario_id: int, emotion: str, agent_type: str, run_ts: str) -> bool:
    return (output_dir / pair_filename(scenario_id, emotion, agent_type, run_ts)).exists()


def write_pair(output_dir: Path, record: dict, agent_type: str, run_ts: str) -> Path:
    dest = output_dir / pair_filename(
        record["scenario_id"], record["target_emotion"], agent_type, run_ts
    )
    dest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


# ── Ollama server helpers ──────────────────────────────────────────────────────

def _ollama_is_running(base_url: str) -> bool:
    try:
        requests.get(f"{base_url}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def check_models(base_url: str) -> None:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
        _log(f"Available models: {available}")
        for needed in [USER_MODEL, CHATBOT_MODEL]:
            if not any(needed in m for m in available):
                _log(f"WARN: Model '{needed}' not listed in Ollama.")
    except Exception as e:
        _log(f"WARN: Could not list models: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline v4: simulate_conversation.py flow + simulate_baseline.py counselor prompt. "
            "Work list loaded from --manifest-dir to match sft-fin corpus pairs exactly."
        )
    )
    parser.add_argument("--xlsx",         default=DEFAULT_XLSX,         help="Path to scenario Excel file")
    parser.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR,   help="Output directory")
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR, help="Directory of reference JSON files (simulated-sft-fin)")
    parser.add_argument("--scenario-ids", nargs="*", type=int,          help="Filter to specific scenario IDs from the manifest")
    parser.add_argument("--ollama-url",   default=DEFAULT_OLLAMA_URL,   help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--resume",       action="store_true",          help="Skip pairs already written in this run's timestamp")
    parser.add_argument("--dry-run",      action="store_true",          help="Print plan without calling Ollama")
    args = parser.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")

    # ── Load scenario details from XLSX ───────────────────────────────────────
    xlsx_path = Path(args.xlsx).resolve()
    if not xlsx_path.exists():
        sys.exit(f"[ERROR] Scenario file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    required_cols = {"ID_skenario", "Kategori", "Topik", "Deskripsi"}
    if not required_cols.issubset(df.columns):
        sys.exit(f"[ERROR] Missing columns. Expected {required_cols}, got {set(df.columns)}")

    scenario_by_id: dict[int, dict] = {
        int(row["ID_skenario"]): row.to_dict()
        for _, row in df.iterrows()
    }

    # ── Load manifest (scenario_id, emotion, agent_type) triples ──────────────
    manifest_dir = Path(args.manifest_dir).resolve()
    if not manifest_dir.is_dir():
        sys.exit(f"[ERROR] Manifest directory not found: {manifest_dir}")

    manifest = load_manifest(manifest_dir)
    if not manifest:
        sys.exit(f"[ERROR] No JSON files found in manifest dir: {manifest_dir}")

    _log(f"Manifest: {len(manifest)} unique (scenario, emotion, agent_type) triples from {manifest_dir.name}/")

    # ── Build work list ────────────────────────────────────────────────────────
    work_list: list[tuple[dict, str, str]] = []
    missing_scenarios: list[int] = []

    for sid, emotion, agent_type in manifest:
        if args.scenario_ids and sid not in args.scenario_ids:
            continue
        if sid not in scenario_by_id:
            missing_scenarios.append(sid)
            continue
        work_list.append((scenario_by_id[sid], emotion, agent_type))

    if missing_scenarios:
        _log(f"WARN: scenario IDs in manifest but not in XLSX: {sorted(set(missing_scenarios))}")

    # ── Output directory ───────────────────────────────────────────────────────
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: drop pairs already written ────────────────────────────────────
    if args.resume:
        before = len(work_list)
        work_list = [
            (s, e, at) for s, e, at in work_list
            if not pair_is_done(output_dir, int(s["ID_skenario"]), e, at, run_ts)
        ]
        skipped = before - len(work_list)
        if skipped:
            _log(f"Resume: skipping {skipped} already-completed pair(s) for run {run_ts}.")

    if args.dry_run:
        _log(f"DRY-RUN — {len(work_list)} pair(s) would run (run_ts={run_ts}):")
        for s, e, at in work_list:
            num_turns = random.randint(TURNS_MIN, TURNS_MAX)
            fname     = pair_filename(int(s["ID_skenario"]), e, at, run_ts)
            print(f"  {fname}  turns={num_turns}  agent={at:<12s}  |  {s['Topik']} | emotion={e}")
        return

    if not _ollama_is_running(args.ollama_url):
        sys.exit(f"[ERROR] Ollama not reachable at {args.ollama_url}. Start it before running.")

    check_models(args.ollama_url)

    _log(f"Starting baseline v4 simulation: {len(work_list)} pair(s), {TURNS_MIN}–{TURNS_MAX} turns each (random per pair).")
    _log(f"Run timestamp: {run_ts}")
    _log(f"User model   : {USER_MODEL}")
    _log(f"Chatbot model: {CHATBOT_MODEL}")
    _log(f"Output dir   : {output_dir}")

    written = 0
    failed  = 0
    t0      = time.time()

    for idx, (scenario, emotion, agent_type) in enumerate(work_list, start=1):
        sid       = int(scenario["ID_skenario"])
        num_turns = random.randint(TURNS_MIN, TURNS_MAX)
        fname     = pair_filename(sid, emotion, agent_type, run_ts)
        _log(f"[{idx}/{len(work_list)}] {fname}  |  {scenario['Topik']} | emotion={emotion} | agent={agent_type} | turns={num_turns}")

        try:
            conv   = simulate_conversation(
                scenario=scenario,
                target_emotion=emotion,
                num_turns=num_turns,
                base_url=args.ollama_url,
                agent_type=agent_type,
            )
            record = conv.to_annotation_ready()
            dest   = write_pair(output_dir, record, agent_type, run_ts)
            written += 1

            elapsed   = time.time() - t0
            remaining = len(work_list) - idx
            rate      = elapsed / idx
            eta_s     = int(rate * remaining)
            eta_str   = f"{eta_s // 3600:02d}h{(eta_s % 3600) // 60:02d}m{eta_s % 60:02d}s"
            _log(f"  OK  -> {dest.name} | written={written} | ETA={eta_str}")

        except Exception as exc:
            failed += 1
            _log(f"  ERROR {fname}: {exc}")

    elapsed_total = int(time.time() - t0)
    _log(
        f"DONE — {written} written, {failed} failed"
        f" | elapsed={elapsed_total // 3600:02d}h"
        f"{(elapsed_total % 3600) // 60:02d}m{elapsed_total % 60:02d}s"
        f" | output-dir={output_dir}"
    )


if __name__ == "__main__":
    main()
