"""
simulate_psydial.py
------------------------------
PsyDial-inspired (arXiv:2408.15787) agent-to-agent simulation baseline.

Adapts the four RMRR principles for Indonesian generation-from-scratch
(not reconstructing real sessions):

  Retrieve  — scenario Deskripsi seeds the chief_complaint.
  Mask      — unknown demographic fields (gender, age, education, family
               relationships) are replaced with [TIDAK DIKETAHUI] in the
               role card. The client model reveals them gradually through
               conversation, staying consistent once disclosed.
  Reconstruct — standard alternating multi-turn loop; client speaks first
               (paper Algorithm 1); default 10 turn-pairs (≈20 utterances).
  Refine    — after each counselor turn, if the response is >50 words,
               contains newlines, or uses numbered lists, a second Ollama
               call rewrites it to 1–2 natural sentences.

LDD metric (Lexical Diversity per Dialogue, from cal_LDD.py):
  PUW  = len(set(tokens)) / len(tokens) × 100   (Proportion of Unique Words)
  UWPD = len(set(tokens)) / num_dialogues        (Unique Words Per Dialogue)
  LDD  = PUW × UWPD
  Tokenization: whitespace split (Indonesian — jieba not used here).
LDD is computed per-dialogue (stored in output JSON) and corpus-wide
(printed as a summary at the end of each run).

Big-Five traits and resistance are constrained per agent_type (same as v5):
  verbose    → Ekstroversi/Keramahan: sedang–tinggi, Neurotisisme: rendah–sedang
  pendiam    → Ekstroversi: rendah, Neurotisisme: sedang–tinggi
  menghindar → Keterbukaan: rendah, Neurotisisme: tinggi
  defensif   → Keramahan: rendah, Neurotisisme: tinggi, Resistensi: tinggi

User prompt : make_client_system_prompt_masked(role_card)  — PsyDial Fig. 7 + Mask note
Counselor   : make_counselor_system_prompt()               — PsyDial Fig. 9
Flow        : client-first (PsyDial Algorithm 1) + Refine step

Output filenames: baseline_psydial_{scenario_id:04d}_{emotion}_{agent_type}_{run_ts}.json
Output dir     : data/simulated-sft_baseline
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
OLLAMA_BIN    = os.getenv("OLLAMA_BIN",    "$HOME/.local/ollama/bin/ollama")
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", "$HOME/.ollama/models")
USER_MODEL    = os.getenv("USER_MODEL",    "gemma4:e4b")
CHATBOT_MODEL = os.getenv("CHATBOT_MODEL", "teta-sft-v2:latest")

DEFAULT_OLLAMA_URL   = "http://localhost:12434"

_SCRIPT_DIR          = Path(__file__).parent
DEFAULT_XLSX         = str(_SCRIPT_DIR / "./data/simulation/skenario_mental.xlsx")
DEFAULT_OUTPUT_DIR   = str(_SCRIPT_DIR / "./results/PsyDial")
DEFAULT_MANIFEST_DIR = str(_SCRIPT_DIR / "./data/simulated-sft_baseline")

# Default target turn-pairs per dialogue (≈20 utterances).
# Double the minimum of other baselines; honours PsyDial's long-horizon design.
TURNS_DEFAULT = 10

# ── Refinement criteria (PsyDial §3: ≤200 Chinese chars → adapted for Indonesian) ─
_REFINE_MAX_WORDS  = 50
_NUMBERED_RE       = re.compile(r"^\s*(?:\d+[\.\)]|[a-zA-Z][\.\)])\s", re.MULTILINE)

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

# ── Big-Five trait options (from simulate_baseline.py) ────────────────────────
_BIG5_LEVELS = {
    "Keterbukaan": [
        ("rendah", "cenderung memilih rutinitas dan enggan mencoba hal baru"),
        ("sedang", "cukup terbuka terhadap pengalaman dan ide baru"),
        ("tinggi", "sangat antusias dengan pengalaman dan perspektif baru"),
    ],
    "Ketelitian": [
        ("rendah", "cenderung spontan dan kurang terorganisir"),
        ("sedang", "cukup bertanggung jawab dan dapat diandalkan"),
        ("tinggi", "sangat terorganisir, disiplin, dan berorientasi pada tujuan"),
    ],
    "Ekstroversi": [
        ("rendah", "introvert, lebih nyaman sendiri dan dalam kelompok kecil"),
        ("sedang", "cukup nyaman dalam situasi sosial namun juga butuh waktu sendiri"),
        ("tinggi", "sangat aktif secara sosial dan mendapat energi dari interaksi"),
    ],
    "Keramahan": [
        ("rendah", "cenderung bersikap skeptis dan menjaga jarak dari orang lain"),
        ("sedang", "umumnya kooperatif dan mudah bergaul"),
        ("tinggi", "sangat empatik, mudah percaya, dan senang membantu orang lain"),
    ],
    "Neurotisisme": [
        ("rendah", "umumnya tenang dan stabil secara emosional"),
        ("sedang", "kadang-kadang merasakan stres atau kekhawatiran yang cukup normal"),
        ("tinggi", "mudah cemas, sensitif terhadap stres, dan rentan terhadap emosi negatif"),
    ],
}

_RESISTANCE_OPTIONS = [
    ("rendah", "klien bersedia terbuka dan berbagi masalahnya secara aktif"),
    ("sedang", "klien mau berbagi namun mungkin memiliki keengganan terhadap beberapa topik pribadi"),
    ("tinggi", "klien tampak ragu-ragu dan cenderung menahan diri untuk mengungkapkan masalah intinya"),
]

# ── Agent-type → allowed Big-Five / resistance levels ─────────────────────────
_AGENT_TYPE_BIG5_CONSTRAINTS: dict[str, dict[str, list[str]]] = {
    "verbose": {
        "Keterbukaan":  ["sedang", "tinggi"],
        "Ketelitian":   ["sedang", "tinggi"],
        "Ekstroversi":  ["sedang", "tinggi"],
        "Keramahan":    ["sedang", "tinggi"],
        "Neurotisisme": ["rendah", "sedang"],
    },
    "pendiam": {
        "Keterbukaan":  ["rendah", "sedang"],
        "Ketelitian":   ["sedang", "tinggi"],
        "Ekstroversi":  ["rendah"],
        "Keramahan":    ["sedang"],
        "Neurotisisme": ["sedang", "tinggi"],
    },
    "menghindar": {
        "Keterbukaan":  ["rendah"],
        "Ketelitian":   ["rendah", "sedang"],
        "Ekstroversi":  ["sedang"],
        "Keramahan":    ["rendah", "sedang"],
        "Neurotisisme": ["tinggi"],
    },
    "defensif": {
        "Keterbukaan":  ["rendah"],
        "Ketelitian":   ["rendah", "sedang"],
        "Ekstroversi":  ["rendah", "sedang"],
        "Keramahan":    ["rendah"],
        "Neurotisisme": ["tinggi"],
    },
}

_AGENT_TYPE_RESISTANCE: dict[str, list[str]] = {
    "verbose":    ["rendah"],
    "pendiam":    ["sedang"],
    "menghindar": ["sedang", "tinggi"],
    "defensif":   ["tinggi"],
}


def _sample_big5_for_agent(agent_type: str) -> str:
    constraints = _AGENT_TYPE_BIG5_CONSTRAINTS.get(agent_type, {})
    lines = []
    for trait, options in _BIG5_LEVELS.items():
        allowed = constraints.get(trait, [opt[0] for opt in options])
        valid   = [opt for opt in options if opt[0] in allowed] or list(options)
        level, desc = random.choice(valid)
        lines.append(f" - {trait}: {level}, {desc}")
    return "\n".join(lines)


def _sample_resistance_for_agent(agent_type: str) -> str:
    allowed = _AGENT_TYPE_RESISTANCE.get(agent_type, [opt[0] for opt in _RESISTANCE_OPTIONS])
    valid   = [opt for opt in _RESISTANCE_OPTIONS if opt[0] in allowed] or list(_RESISTANCE_OPTIONS)
    level, desc = random.choice(valid)
    return f"{level}, {desc}"


# ── Mask step: role card with [TIDAK DIKETAHUI] placeholders ──────────────────

def make_role_card_masked(
    scenario: dict, target_emotion: str, agent_type: str
) -> tuple[str, str, list[str]]:
    """
    Returns (profile_detail, chief_complaint, masked_fields).

    Known fields: Big-Five (constrained by agent_type), resistance, chief_complaint.
    Masked fields: gender, age, family relationships — replaced with [TIDAK DIKETAHUI].
    The client model reveals these gradually during conversation (Mask step of RMRR).
    """
    big5       = _sample_big5_for_agent(agent_type)
    resistance = _sample_resistance_for_agent(agent_type)

    masked_fields = ["Jenis kelamin", "Usia", "Hubungan keluarga"]

    profile_detail = (
        "Jenis kelamin: [TIDAK DIKETAHUI]\n"
        "Usia: [TIDAK DIKETAHUI]\n"
        "Jenjang pendidikan: SMA\n"
        "Status/Pekerjaan: Pelajar\n"
        "Status pernikahan: Belum menikah\n"
        "Hubungan keluarga: [TIDAK DIKETAHUI]\n"
        f"Kepribadian Lima Besar:\n{big5}\n"
        f"Tingkat resistensi terhadap konseling: {resistance}"
    )

    chief_complaint = (
        f"Masalah utama: {scenario['Deskripsi']}\n"
        f"Kategori masalah: {scenario['Kategori']} — {scenario['Topik']}\n"
        f"Kondisi emosional saat ini: {target_emotion}"
    )

    return profile_detail, chief_complaint, masked_fields


# ── System prompts ─────────────────────────────────────────────────────────────

def make_client_system_prompt_masked(role_card: str) -> str:
    """
    PsyDial Figure 7 client prompt, adapted for Indonesian with Mask instruction.
    Fields marked [TIDAK DIKETAHUI] are revealed gradually through conversation.
    """
    return f"""Sekarang kamu adalah seorang siswa SMA yang mengikuti sesi konseling psikologi.
Berikut adalah informasi pribadimu:
{role_card}

Catatan untuk bidang bertanda [TIDAK DIKETAHUI]: kamu menyimpan detail tersebut untuk diri sendiri.
Ungkapkan secara bertahap selama percakapan berlangsung sesuai dengan kebutuhan dan kenyamananmu,
dan pastikan kamu konsisten dengan detail yang sudah kamu ungkapkan sebelumnya.

Persyaratan dialog untuk klien:
1. Berdasarkan masalah utama yang kamu ajukan, ekspresi harus sesuai dengan gaya bicara klien, senatural dan sesehari-hari mungkin.
2. Jawab hanya berdasarkan informasi pribadi, tetap setia pada informasi pribadi.
3. Kamu harus menguraikan masalahmu secara bertahap dan menceritakannya kepada konselor sedikit demi sedikit.
4. Setiap kali berbicara batasi 1 hingga 2 kalimat, pertahankan peranmu saat berbicara.
5. Jangan terlalu cepat mengucapkan "terima kasih" atau "selamat tinggal".
6. Proses konseling berlangsung selama beberapa putaran interaksi."""


def make_counselor_system_prompt() -> str:
    """
    PsyDial Figure 9 counselor prompt — 3-phase counseling, 12-point dialog rules.
    Identical to simulate_baseline.py and simulate_baseline_v5.py.
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


# ── Refine step ────────────────────────────────────────────────────────────────

_REFINE_SYSTEM = """Kamu adalah editor respons konselor psikologi.
Tulis ulang respons berikut agar:
1. Tidak melebihi 2 kalimat pendek (di bawah 50 kata).
2. Tidak mengandung baris baru, daftar bernomor, atau poin-poin.
3. Tetap alami, empatik, dan relevan dengan konteks percakapan.
Tulis HANYA teks respons yang sudah diperbaiki tanpa penjelasan tambahan."""


def _needs_refinement(text: str) -> bool:
    """Return True if the counselor response fails PsyDial's refinement criteria."""
    return (
        len(text.split()) > _REFINE_MAX_WORDS
        or "\n" in text
        or bool(_NUMBERED_RE.search(text))
    )


def _refine_response(response: str, base_url: str) -> str:
    """Rewrite an overly-long or poorly-structured counselor response (Refine step)."""
    return ollama_chat(
        model=CHATBOT_MODEL,
        messages=[{"role": "user", "content": f"Respons yang perlu diperbaiki:\n{response}"}],
        system=_REFINE_SYSTEM,
        base_url=base_url,
        temperature=0.5,
    )


# ── LDD metric (from PsyDial cal_LDD.py, adapted for Indonesian) ──────────────

def compute_ldd(records: list[dict]) -> dict[str, dict]:
    """
    Corpus-level LDD over a list of annotation-ready records.

      PUW  = len(set(tokens)) / len(tokens) × 100
      UWPD = len(set(tokens)) / num_dialogues
      LDD  = PUW × UWPD

    Tokenization: whitespace split (Indonesian — not jieba).
    Mirrors cal_LDD.py: separate metrics for client and counselor sides.
    """
    n = len(records)
    if n == 0:
        return {}

    client_tokens: list[str]   = []
    counselor_tokens: list[str] = []

    for rec in records:
        for msg in rec.get("messages", []):
            tokens = msg["content"].lower().split()
            if msg["role"] == "user":
                client_tokens.extend(tokens)
            else:
                counselor_tokens.extend(tokens)

    result: dict[str, dict] = {}
    for role, tokens in [("client", client_tokens), ("counselor", counselor_tokens)]:
        if not tokens:
            result[role] = {"PUW": 0.0, "UWPD": 0.0, "LDD": 0.0,
                            "total_tokens": 0, "unique_tokens": 0}
            continue
        puw  = round(len(set(tokens)) / len(tokens) * 100, 2)
        uwpd = round(len(set(tokens)) / n, 2)
        result[role] = {
            "PUW":           puw,
            "UWPD":          uwpd,
            "LDD":           round(puw * uwpd, 2),
            "total_tokens":  len(tokens),
            "unique_tokens": len(set(tokens)),
        }
    return result


def compute_ldd_single(record: dict) -> dict[str, dict]:
    """Per-dialogue LDD (corpus of 1). Stored in output JSON metadata."""
    return compute_ldd([record])


# ── Closing-message detectors ──────────────────────────────────────────────────

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
    """Return True if the client message looks like a farewell/closing turn."""
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


# ── Action/thinking sanitizer ──────────────────────────────────────────────────

_ACTION_PATTERN = re.compile(
    r"\(([^)]*)\)"    # (...)
    r"|\[([^\]]*)\]"  # [...] — but not [TIDAK DIKETAHUI] (already resolved in convo)
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
    role: str        # "user" (client) | "assistant" (counselor)
    content: str
    turn_index: int
    refined: bool = False


@dataclass
class SimulatedConversation:
    dialog_id: str
    scenario_id: int
    kategori: str
    topik: str
    deskripsi: str
    target_emotion: str
    profile_detail: str
    chief_complaint: str
    masked_fields: list[str]
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
        record = {
            "dialog_id":       self.dialog_id,
            "scenario_id":     self.scenario_id,
            "kategori":        self.kategori,
            "topik":           self.topik,
            "deskripsi":       self.deskripsi,
            "target_emotion":  self.target_emotion,
            "profile_detail":  self.profile_detail,
            "chief_complaint": self.chief_complaint,
            "masked_fields":   self.masked_fields,
            "messages":        messages,
            "dialogue":        dialogue_text,
            "turns_raw":       turns_dict,
            "metadata":        self.metadata,
        }
        record["ldd"] = compute_ldd_single(record)
        return record


# ── Core simulation (RMRR: Reconstruct loop + Refine step) ────────────────────

def simulate_conversation(
    scenario: dict,
    target_emotion: str,
    num_turns: int,
    base_url: str,
    agent_type: str = "verbose",
) -> SimulatedConversation:
    """
    PsyDial-inspired flow (client-first, paper Algorithm 1):
      Turn 0: client opens with a natural greeting (Reconstruct begins).
      Each odd turn: counselor responds; validated and refined if needed (Refine).
      Terminates on client farewell, counselor closing phrase, or num_turns reached.

    Role card uses [TIDAK DIKETAHUI] for unknown demographics (Mask step).
    Scenario Deskripsi seeds chief_complaint (Retrieve step).
    """
    scenario_id = int(scenario["ID_skenario"])
    dialog_id   = (
        f"baseline_b_psydial_{scenario_id:04d}_{target_emotion}_{int(time.time())}"
    )

    profile_detail, chief_complaint, masked_fields = make_role_card_masked(
        scenario, target_emotion, agent_type
    )
    full_role_card = f"{profile_detail}\n{chief_complaint}"

    client_sys    = make_client_system_prompt_masked(full_role_card)
    counselor_sys = make_counselor_system_prompt()

    conv = SimulatedConversation(
        dialog_id=dialog_id,
        scenario_id=scenario_id,
        kategori=scenario["Kategori"],
        topik=scenario["Topik"],
        deskripsi=scenario["Deskripsi"],
        target_emotion=target_emotion,
        profile_detail=profile_detail,
        chief_complaint=chief_complaint,
        masked_fields=masked_fields,
        metadata={
            "user_model":       USER_MODEL,
            "chatbot_model":    CHATBOT_MODEL,
            "agent_type":       agent_type,
            "num_turn_pairs":   num_turns,
            "simulated_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "baseline":         "psydial-arXiv-2408.15787-rmrr-adapt-b",
            "refine_max_words": _REFINE_MAX_WORDS,
        },
    )

    # shared_history in counselor perspective: client→"user", counselor→"assistant"
    shared_history: list[dict] = []
    refinement_count = 0

    for pair_idx in range(num_turns):
        # ── Client turn (Reconstruct: client speaks first) ─────────────────────
        if pair_idx == 0:
            client_messages = [
                {
                    "role": "user",
                    "content": (
                        "Mulailah sesi konseling. "
                        "Buka dengan salam singkat yang natural, "
                        "seperti menyapa konselor untuk pertama kalinya."
                    ),
                }
            ]
        else:
            # Reverse shared_history for client model view
            client_messages = []
            for msg in shared_history:
                if msg["role"] == "user":
                    client_messages.append({"role": "assistant", "content": msg["content"]})
                else:
                    client_messages.append({"role": "user", "content": msg["content"]})

        client_reply = _strip_actions(ollama_chat(
            model=USER_MODEL,
            messages=client_messages,
            system=client_sys,
            base_url=base_url,
            temperature=0.9,
        ))
        turn_idx_u = pair_idx * 2

        if not client_reply:
            _log(f"    [EMPTY] Client reply empty at turn {turn_idx_u} — ending.")
            bot_closing = (
                "Oke, sepertinya kamu butuh waktu. "
                "Kalau mau cerita lagi, Teta selalu di sini ya. "
                "Jaga diri kamu baik-baik."
            )
            conv.turns.append(Turn(
                role="assistant", content=bot_closing,
                turn_index=turn_idx_u + 1,
            ))
            _log(f"    [B{turn_idx_u + 1}] {bot_closing}")
            break

        shared_history.append({"role": "user", "content": client_reply})
        conv.turns.append(Turn(role="user", content=client_reply, turn_index=turn_idx_u))
        _log(f"    [U{turn_idx_u}] {client_reply[:100].replace(chr(10), ' ')}...")

        client_farewell = _is_closing_message(client_reply)

        # ── Counselor turn ─────────────────────────────────────────────────────
        bot_reply = _strip_actions(ollama_chat(
            model=CHATBOT_MODEL,
            messages=shared_history,
            system=counselor_sys,
            base_url=base_url,
            temperature=0.75,
        ))

        # ── Refine step: rewrite if response fails quality criteria ───────────
        refined = False
        if _needs_refinement(bot_reply):
            bot_reply_refined = _strip_actions(_refine_response(bot_reply, base_url))
            if bot_reply_refined:
                bot_reply = bot_reply_refined
                refined = True
                refinement_count += 1

        shared_history.append({"role": "assistant", "content": bot_reply})
        turn_idx_b = pair_idx * 2 + 1
        conv.turns.append(Turn(
            role="assistant", content=bot_reply,
            turn_index=turn_idx_b, refined=refined,
        ))
        refine_tag = " [refined]" if refined else ""
        _log(f"    [B{turn_idx_b}]{refine_tag} {bot_reply[:100].replace(chr(10), ' ')}...")

        if client_farewell:
            _log(f"    [CLOSING] Client farewell at turn {turn_idx_u} — ending.")
            break
        if _counselor_is_closing(bot_reply):
            _log(f"    [CLOSING] Counselor closing phrase at turn {turn_idx_b} — ending.")
            break

    conv.metadata["refinement_count"] = refinement_count
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
    return f"baseline_b_psydial_{scenario_id:04d}_{safe_emotion}_{agent_type}_{run_ts}.json"


def pair_is_done(
    output_dir: Path, scenario_id: int, emotion: str, agent_type: str, run_ts: str
) -> bool:
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
            "Baseline B (PsyDial-inspired RMRR): client-first simulation with masked "
            "role card, Refine step for counselor responses, and LDD metric output. "
            "Work list from --manifest-dir matches sft-fin corpus pairs exactly."
        )
    )
    parser.add_argument("--xlsx",         default=DEFAULT_XLSX,         help="Path to scenario Excel file")
    parser.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR,   help="Output directory")
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR, help="Directory of reference JSON files (simulated-sft-fin)")
    parser.add_argument("--scenario-ids", nargs="*", type=int,          help="Filter to specific scenario IDs from the manifest")
    parser.add_argument("--ollama-url",   default=DEFAULT_OLLAMA_URL,   help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--turns",        type=int, default=TURNS_DEFAULT,
                        help=f"Target turn-pairs per dialogue (default: {TURNS_DEFAULT})")
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
            fname = pair_filename(int(s["ID_skenario"]), e, at, run_ts)
            print(f"  {fname}  turns={args.turns}  agent={at:<12s}  |  {s['Topik']} | emotion={e}")
        return

    if not _ollama_is_running(args.ollama_url):
        sys.exit(f"[ERROR] Ollama not reachable at {args.ollama_url}. Start it before running.")

    check_models(args.ollama_url)

    _log(f"Starting baseline B (PsyDial-inspired RMRR): {len(work_list)} pair(s), {args.turns} turn-pairs each.")
    _log(f"Run timestamp: {run_ts}")
    _log(f"User model   : {USER_MODEL}")
    _log(f"Chatbot model: {CHATBOT_MODEL}")
    _log(f"Refine step  : enabled (>{_REFINE_MAX_WORDS} words | newlines | numbered lists)")
    _log(f"Output dir   : {output_dir}")

    written  = 0
    failed   = 0
    all_recs: list[dict] = []
    t0       = time.time()

    for idx, (scenario, emotion, agent_type) in enumerate(work_list, start=1):
        sid   = int(scenario["ID_skenario"])
        fname = pair_filename(sid, emotion, agent_type, run_ts)
        _log(
            f"[{idx}/{len(work_list)}] {fname}"
            f"  |  {scenario['Topik']} | emotion={emotion} | agent={agent_type}"
            f" | turns={args.turns}"
        )

        try:
            conv   = simulate_conversation(
                scenario=scenario,
                target_emotion=emotion,
                num_turns=args.turns,
                base_url=args.ollama_url,
                agent_type=agent_type,
            )
            record = conv.to_annotation_ready()
            dest   = write_pair(output_dir, record, agent_type, run_ts)
            all_recs.append(record)
            written += 1

            per_ldd = record.get("ldd", {})
            elapsed   = time.time() - t0
            remaining = len(work_list) - idx
            rate      = elapsed / idx
            eta_s     = int(rate * remaining)
            eta_str   = f"{eta_s // 3600:02d}h{(eta_s % 3600) // 60:02d}m{eta_s % 60:02d}s"
            refinements = record.get("metadata", {}).get("refinement_count", 0)
            _log(
                f"  OK  -> {dest.name}"
                f" | written={written}"
                f" | refined={refinements}"
                f" | LDD_client={per_ldd.get('client', {}).get('LDD', 'n/a')}"
                f" | ETA={eta_str}"
            )

        except Exception as exc:
            failed += 1
            _log(f"  ERROR {fname}: {exc}")

    # ── Corpus-level LDD summary ───────────────────────────────────────────────
    if all_recs:
        corpus_ldd = compute_ldd(all_recs)
        _log("─" * 60)
        _log(f"Corpus LDD summary ({len(all_recs)} dialogues):")
        for role, stats in corpus_ldd.items():
            _log(
                f"  {role:<10s}  PUW={stats['PUW']:.2f}%"
                f"  UWPD={stats['UWPD']:.2f}"
                f"  LDD={stats['LDD']:.2f}"
                f"  (total_tokens={stats['total_tokens']}, unique={stats['unique_tokens']})"
            )
        _log("─" * 60)

    elapsed_total = int(time.time() - t0)
    _log(
        f"DONE — {written} written, {failed} failed"
        f" | elapsed={elapsed_total // 3600:02d}h"
        f"{(elapsed_total % 3600) // 60:02d}m{elapsed_total % 60:02d}s"
        f" | output-dir={output_dir}"
    )


if __name__ == "__main__":
    main()
