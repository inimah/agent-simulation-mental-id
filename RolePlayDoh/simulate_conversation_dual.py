"""
simulate_conversation_dual.py  —  Baseline D: Roleplay-doh Dual Adherence with AI-awareness
---------------------------------------------------------------------------
Extends Baseline (Roleplay-doh) by applying principle-adherence prompting
to BOTH agents in every turn:

  USER AGENT   — same Roleplay-doh patient principles as Baseline D
                 (8 principles: 6 agent-type + 2 category-specific)
  CHATBOT AGENT — NEW: stage-aware counseling principle reminder injected
                 into the chatbot system prompt before every chatbot turn,
                 derived from "kaidah psikologi untuk remaja di Indonesia"
                 (CBT, person-centered, trauma-informed, cultural empathy)

This isolates the effect of chatbot adherence prompting over user-only
adherence (Baseline) for fair ablation comparison.

Output filenames : roleplay_doh_dual_{scenario_id:04d}_{emotion}_{agent_type}_{run_ts}.json
Output directory : ./results/RolePlayDoh-aiaware

Usage
-----
    python simulate_conversation_dual.py \\
        --xlsx         ./data/simulation/skenario_mental.xlsx \\
        --output-dir   ./results/RolePlayDoh-aiaware \\
        [--manifest    ./data/simulated-sft_baseline] \\
        [--scenario-ids 1 5 12] \\
        [--agent-type pendiam] \\
        [--emotions-per-scenario 1] \\
        [--turns 10] \\
        [--resume] [--dry-run] \\
        [--ollama-url http://localhost:12434]
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
OLLAMA_BIN    = os.getenv("OLLAMA_BIN",    "$HOME/.local/ollama/bin/ollama")
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", "$HOME/.ollama/models")
USER_MODEL    = os.getenv("USER_MODEL",    "gemma4:e4b")
CHATBOT_MODEL = os.getenv("CHATBOT_MODEL", "teta-sft-v2:latest")

DEFAULT_OLLAMA_URL = "http://localhost:12434"

_SCRIPT_DIR        = Path(__file__).parent
DEFAULT_XLSX       = str(_SCRIPT_DIR / "../data/simulation/skenario_mental.xlsx")
DEFAULT_OUTPUT_DIR = str(_SCRIPT_DIR / "./results/RolePlayDoh-aiaware")

TURNS_DEFAULT = 25
TURNS_MIN     = 15
TURNS_MAX     = 25

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

USER_AGENT_TYPES = ["verbose", "pendiam", "menghindar", "defensif"]


# ── Roleplay-doh: USER AGENT principle sets ───────────────────────────────────

_AGENT_PRINCIPLES: dict[str, list[str]] = {
    "pendiam": [
        "Berikan respons sangat singkat — 1–2 kalimat pendek, sering hanya satu frasa atau kata",
        "Tunjukkan keengganan awal yang nyata; butuh minimal 3 giliran sebelum mengungkapkan inti masalah",
        "Jangan sukarela memberikan informasi baru; ungkapkan detail hanya jika ditanya secara langsung",
        "Ekspresikan emosi sangat tidak langsung: melalui nada dan pilihan kata, bukan pernyataan eksplisit",
        "Tunjukkan ketidakyakinan bahwa bercerita akan membantu ('ya mau gimana juga', 'nggak tau juga')",
        "Mulai terbuka sedikit demi sedikit hanya setelah Teta menunjukkan kesabaran dan empati yang konsisten",
    ],
    "menghindar": [
        "Alihkan ke topik lain atau buat humor ringan ketika pertanyaan menyentuh sisi personal yang menyakitkan",
        "Gunakan frasa pengalih yang khas: 'ya udahlah', 'gapapa kok', 'nggak penting juga sih'",
        "Minimalisasi masalah saat dipertanyakan lebih dalam: 'lebay gue ngomonginnya', 'biasa aja sih'",
        "Ungkap inti masalah hanya secara tidak langsung dan bertahap, setelah beberapa giliran penghindaran",
        "Tunjukkan sebentar jejak emosi nyata lalu segera tutupi dengan pengalihan atau humor ringan",
        "Tunjukkan rasa lega ketika Teta memberi ruang tanpa memaksa — baru waktu itu sedikit terbuka",
    ],
    "defensif": [
        "Tunjukkan skeptisisme awal terhadap kemampuan chatbot: 'kamu bot, mana bisa ngerti'",
        "Tolak atau pertanyakan saran yang datang terlalu cepat: 'itu mah gampang ngomongnya'",
        "Ungkapkan frustrasi jika merasa tidak dipahami atau digeneralisasi oleh Teta",
        "Pertanyakan relevansi atau tujuan pertanyaan Teta sesekali ('emang nanya itu buat apa?')",
        "Butuh bukti empati yang genuine dan spesifik sebelum menerima sudut pandang baru",
        "Tunjukkan keterbukaan kecil — hanya setelah Teta menunjukkan pemahaman yang tepat dan mendalam",
    ],
    "verbose": [
        "Berikan respons 2–4 kalimat dengan detail yang cukup tentang situasimu",
        "Gunakan bahasa remaja yang natural; boleh campur sedikit kata Inggris atau slang Indonesia",
        "Ekspresikan emosi lebih terbuka dari tipe lain, tapi tetap implisit — bukan pernyataan langsung",
        "Reaksi terhadap Teta lebih langsung: tunjukkan apakah kamu merasa didengar atau tidak",
        "Sesekali muncul keraguan sendiri dalam responsmu ('ya tapi ga tau juga sih...', 'hmm')",
        "Tunjukkan pergeseran emosi yang nyata saat Teta berhasil menggali lebih dalam masalahmu",
    ],
}

_CATEGORY_PRINCIPLES: dict[str, list[str]] = {
    "Konsep identitas diri": [
        "Ekspresikan kebingungan tentang identitas diri secara tidak linear — pemikiranmu melompat-lompat",
        "Hindari label diri yang jelas; gunakan kata-kata seperti 'aneh', 'beda', 'ga tau' daripada istilah klinis",
    ],
    "Pengendalian impuls": [
        "Ceritakan tindakan impulsif dengan campuran penyesalan dan pembelaan diri sendiri",
        "Tunjukkan kesulitan menjelaskan mengapa kamu bertindak seperti itu — 'ya tiba-tiba aja'",
    ],
    "Hubungan Interpersonal": [
        "Ekspresikan konflik hubungan dari perspektif yang campur antara menyalahkan orang lain dan dirimu sendiri",
        "Tunjukkan ketakutan akan penolakan atau kehilangan hubungan penting itu",
    ],
    "Stress Akademik": [
        "Bandingkan dirimu dengan teman atau ekspektasi orang tua/guru secara berulang dalam ceritamu",
        "Tunjukkan kelelahan kumulatif yang menumpuk, bukan hanya stres satu momen tertentu",
    ],
    "Kecemasan Sosial": [
        "Tunjukkan kekhawatiran tentang penilaian orang lain bahkan dalam percakapan ini dengan Teta",
        "Gunakan koreksi diri sendiri: mulai kalimat lalu ubah arah ('maksudnya... ya ga tau deh')",
    ],
    "ADHD": [
        "Ganti topik secara tidak terduga di tengah kalimat; alurmu tidak linear",
        "Tunjukkan frustrasi dengan dirimu sendiri karena tidak bisa fokus atau menjelaskan dengan runtut",
    ],
    "Eating Disorder": [
        "Gunakan bahasa eufemistik untuk hubunganmu dengan tubuh/makanan ('kebiasaan aku', 'cara aku cope')",
        "Tunjukkan ambivalensi nyata: ingin berubah tapi takut kehilangan rasa kontrol yang kamu miliki",
    ],
    "Primary Support": [
        "Ekspresikan kehilangan atau perubahan melalui detail kecil dan spesifik, bukan pernyataan besar",
        "Tunjukkan beban tak terucap dari selalu menjaga atau memperhatikan orang lain di atas dirimu sendiri",
    ],
    "Non-Suicidal Self-Injury": [
        "Gunakan eufemisme untuk self-harm; hindari menyebutnya secara langsung ('hal yang aku lakuin')",
        "Tunjukkan ambivalensi: ada bagian yang ingin berhenti, ada bagian yang belum siap melepasnya",
    ],
    "Self Harm": [
        "Ekspresikan krisis dengan cara yang fragmentaris dan tersirat, bukan langsung",
        "Tunjukkan keengganan kuat untuk memberikan detail spesifik — resistensi nyata terhadap pengungkapan",
    ],
    "Perilaku Beresiko": [
        "Ekspresikan perilaku berisiko dengan campuran sikap bangga (di depan teman) dan malu (sekarang)",
        "Tunjukkan normalisasi perilaku tersebut di lingkungan pertemananmu sebagai konteks",
    ],
}


def _build_principles(scenario: dict, agent_type: str) -> list[str]:
    """Combine agent-type base principles with category-specific principles."""
    base     = list(_AGENT_PRINCIPLES.get(agent_type, _AGENT_PRINCIPLES["verbose"]))
    category = scenario.get("Kategori", "")
    extra    = _CATEGORY_PRINCIPLES.get(category, [])
    return base + extra


# ── Roleplay-doh: CHATBOT AGENT principle sets (NEW in Baseline D2) ───────────
# Kaidah psikologi konseling untuk remaja Indonesia:
# pendekatan CBT, person-centered, dan trauma-informed dengan penekanan pada
# empati genuine, cultural understanding, dan komunikasi yang setara.

_CHATBOT_PRINCIPLES: list[str] = [
    "Validasi perasaan remaja dengan refleksi emosi yang SPESIFIK dan terasa genuine sebelum memberi respons apapun — hindari frasa validasi generik yang terasa copy-paste",
    "Tahan diri dari saran atau solusi prematur; gali lebih dalam dulu dengan pertanyaan terbuka yang relevan dengan situasi spesifik mereka",
    "Gunakan bahasa setara dan hangat yang sesuai untuk remaja Indonesia — hindari nada menggurui, menghakimi, atau seperti orang tua berbicara ke anak",
    "Hormati konteks budaya kolektif Indonesia: tekanan keluarga, ekspektasi sosial, dan rasa malu (malu) adalah faktor nyata yang membentuk pengalaman mereka — akui keberadaannya, jangan abaikan atau minimalisasi",
    "Deteksi distress yang tersembunyi di balik kata-kata tidak langsung — remaja Indonesia sering menyampaikan masalah serius melalui eufemisme, humor, atau narasi minimal",
    "Jika remaja menutup diri, mengalihkan topik, atau menunjukkan resistensi, beri ruang dan coba pendekatan yang lebih lembut — jangan memaksa pengungkapan",
]

_CHATBOT_STAGE_FOCUS: dict[str, dict] = {
    "AWAL": {
        "label": "AWAL PERCAKAPAN — Bangun Rapport & Rasa Aman",
        "focus": (
            "Prioritasmu sekarang adalah membuat remaja merasa AMAN dan DIDENGAR, bukan menggali informasi. "
            "Refleksikan apa yang sudah mereka ungkapkan dengan spesifik. "
            "Hindari pertanyaan yang terlalu langsung atau personal di tahap awal ini. "
            "Tunjukkan bahwa kamu hadir sepenuhnya — bukan sekadar menjalankan skrip."
        ),
        "key_principles": [0, 2, 4, 5],
    },
    "TENGAH": {
        "label": "TENGAH PERCAKAPAN — Eksplorasi Mendalam & Empati Kontekstual",
        "focus": (
            "Remaja sudah mulai terbuka. Gunakan detail spesifik yang mereka ceritakan untuk menunjukkan "
            "pemahaman yang mendalam — sebutkan nama orang, situasi, atau perasaan yang sudah mereka sebut. "
            "Lakukan cognitive reframing yang lembut jika ada pola pikir distortif. "
            "Akui dilema antara keinginan pribadi dan tekanan sosial/keluarga jika relevan. "
            "Gunakan pertanyaan Socratic untuk membuka perspektif baru secara tidak langsung."
        ),
        "key_principles": [0, 1, 3, 4],
    },
    "AKHIR": {
        "label": "AKHIR PERCAKAPAN — Penutupan Bermakna & Penguatan",
        "focus": (
            "Dukung penutupan yang bermakna dan tidak tergesa-gesa. "
            "Berikan penguatan positif yang spesifik atas keberanian mereka membuka diri. "
            "Tawarkan langkah konkret atau sumber daya HANYA jika relevan dan mereka mau menerimanya. "
            "Biarkan remaja yang memimpin penutupan — ikuti arahnya, jangan paksa agenda sendiri. "
            "Jika ada sinyal krisis, sampaikan Into The Light Indonesia 119 ext 8 dengan cara yang lembut."
        ),
        "key_principles": [0, 2, 3, 5],
    },
}


def _make_chatbot_adherence_reminder(pair_idx: int, num_turns: int) -> str:
    """
    Roleplay-doh principle-adherence prompting for the CHATBOT agent.
    Appended to the chatbot system prompt before every chatbot turn.
    Derived from Indonesian adolescent psychology counseling principles:
    CBT, person-centered, trauma-informed, with cultural empathy emphasis.
    """
    if pair_idx == 0 or pair_idx < max(1, int(num_turns * 0.35)):
        stage_key = "AWAL"
    elif pair_idx < int(num_turns * 0.70):
        stage_key = "TENGAH"
    else:
        stage_key = "AKHIR"

    stage_data = _CHATBOT_STAGE_FOCUS[stage_key]
    key_idxs   = stage_data["key_principles"]
    key_items  = "\n".join(
        f"  • {_CHATBOT_PRINCIPLES[i]}" for i in key_idxs
    )

    return (
        f"\n---\n"
        f"[PANDUAN KONSELING — {stage_data['label']} (giliran {pair_idx + 1}/{num_turns})]\n"
        f"{stage_data['focus']}\n\n"
        f"Prinsip kunci untuk momen ini:\n{key_items}\n\n"
        f"Terapkan prinsip-prinsip ini secara alami dalam responsmu. "
        f"Responsmu harus terasa seperti konselor yang bijak dan empatik, bukan skrip. "
        f"Sekarang balas sebagai Teta:"
    )


# ── Roleplay-doh: Patient profile ─────────────────────────────────────────────

def _build_backstory(scenario: dict, target_emotion: str) -> str:
    return (
        f"Kamu adalah seorang siswa SMA berusia 16–17 tahun. "
        f"Kategori masalahmu: {scenario.get('Kategori', '')} — {scenario.get('Topik', '')}. "
        f"Situasi yang kamu alami: {scenario.get('Deskripsi', '')} "
        f"Saat ini kamu merasakan emosi: {target_emotion}. "
        "Kamu memutuskan untuk membuka aplikasi Teta karena ingin curhat atau mencari dukungan — "
        "kamu tahu Teta adalah chatbot AI, bukan manusia."
    )


def make_user_system_prompt(
    scenario: dict,
    target_emotion: str,
    agent_type: str,
    principles: list[str],
) -> str:
    backstory       = _build_backstory(scenario, target_emotion)
    principles_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(principles))

    style_note = {
        "pendiam":    "Singkat dan pendiam. Butuh banyak dorongan sebelum mau cerita.",
        "menghindar": "Menghindari topik personal. Sering bercanda atau ganti topik.",
        "defensif":   "Skeptis dan mudah defensif. Perlu meyakinkan dulu sebelum terbuka.",
        "verbose":    "Cukup terbuka. Berbicara natural dengan detail yang memadai.",
    }.get(agent_type, "Natural dan jujur.")

    return f"""Kamu adalah seorang siswa SMA yang membuka aplikasi chatbot kesehatan mental Teta di ponselmu.

LATAR BELAKANG DIRIMU:
{backstory}

MASALAH YANG KAMU BAWA KE TETA:
Kategori : {scenario.get('Kategori', '')}
Topik    : {scenario.get('Topik', '')}
Situasi  : {scenario.get('Deskripsi', '')}
Emosi awal: {target_emotion}

PRINSIP YANG MENGATUR PERILAKUMU SELAMA PERCAKAPAN INI:
(Roleplay-doh: ikuti SEMUA prinsip ini — prinsip-prinsip ini mendefinisikan bagaimana kamu berperilaku)
{principles_text}

GAYA KOMUNIKASI UMUM: {style_note}

ATURAN TEKNIS:
1. Tulis HANYA teks pesan yang kamu ketik — murni teks percakapan.
2. Gunakan bahasa remaja Indonesia yang natural (boleh campur sedikit Inggris/slang).
3. Tunjukkan emosi {target_emotion} secara tersirat melalui pilihan kata dan nada — JANGAN nyatakan langsung.
4. Bereaksi secara realistis terhadap respons Teta sesuai prinsip-prinsipmu.
5. Jangan pernah keluar dari peran.
6. Jika percakapan sudah terasa selesai, kirim SATU pesan penutup singkat (misal: "Makasih ya", "Ok deh"). Setelah itu BERHENTI.
7. DILARANG: deskripsi tindakan fisik seperti (Menghela napas), (Diam sebentar), atau tanda kurung apapun.
8. DILARANG: narasi internal atau komentar di luar pesan seperti "aku berpikir..." atau "[dalam hati]".
"""


def _make_principle_adherence_reminder(
    pair_idx: int,
    num_turns: int,
    principles: list[str],
    target_emotion: str,
) -> str:
    if pair_idx == 0:
        stage = "AWAL PERCAKAPAN"
        focus = (
            "Ini giliran pertamamu. Buka topik secara singkat dan natural sesuai situasimu. "
            "Tunjukkan prinsip pembukaan: keengganan awal, gaya komunikasimu, dan emosi awal yang tersirat. "
            f"Emosi awal yang harus tersirat: {target_emotion}."
        )
    elif pair_idx < max(2, int(num_turns * 0.35)):
        stage = "AWAL PERCAKAPAN"
        focus = (
            "Kamu masih di tahap awal. Ungkapkan informasi secara terbatas sesuai prinsip pembukaan. "
            "Reaksi terhadap respons Teta harus mencerminkan prinsip keengganan atau penghindaran awalmu."
        )
    elif pair_idx < int(num_turns * 0.70):
        stage = "TENGAH PERCAKAPAN"
        focus = (
            "Percakapan sedang berkembang. Ungkapkan detail masalahmu secara bertahap sesuai prinsip pengungkapanmu. "
            "Reaksi terhadap empati atau saran Teta harus mencerminkan kepribadianmu secara konsisten."
        )
    else:
        stage = "AKHIR PERCAKAPAN"
        focus = (
            "Percakapan hampir selesai. Tunjukkan apakah kamu merasa cukup didengar atau tidak. "
            "Akhiri secara natural sesuai gaya komunikasimu dan prinsip penutupanmu."
        )

    principles_short = "\n".join(f"  • {p}" for p in principles[:4])
    return (
        f"[PENGINGAT PRINSIP — {stage} (giliran {pair_idx + 1}/{num_turns})]\n"
        f"{focus}\n"
        f"Prinsip utamamu yang relevan sekarang:\n{principles_short}\n"
        f"Tinjau semua prinsip di atas dan pilih yang paling relevan untuk momen ini sebelum merespons.\n"
        "---\n"
        "Sekarang ketik pesanmu:"
    )


# ── Chatbot system prompt ──────────────────────────────────────────────────────

def make_chatbot_system_prompt(
    avoid_phrases: set[str] | None = None,
    first_turn: bool = True,
) -> str:
    avoid_block = ""
    if avoid_phrases:
        phrase_list = ", ".join(f'"{p}"' for p in sorted(avoid_phrases))
        avoid_block = (
            f"\n13. DILARANG mengulang frasa validasi yang sudah dipakai sebelumnya: {phrase_list}. "
            "Gunakan cara lain yang segar untuk menunjukkan empati."
        )
    intro_rule = (
        "Kamu boleh memperkenalkan diri sebagai Teta hanya pada giliran pertamamu."
        if first_turn else
        "DILARANG memperkenalkan diri lagi — jangan ulangi 'Hai, aku Teta' atau frasa serupa."
    )
    return f"""Kamu adalah Teta, chatbot kesehatan mental untuk remaja. Kamu adalah AI — bukan manusia. {intro_rule}

Pendekatanmu mengikuti prinsip CBT, person-centered, dan trauma-informed.

PANDUAN:
1. Validasi perasaan pengguna SEBELUM memberi saran apapun.
2. Gunakan teknik active listening: refleksi, parafrase, pertanyaan terbuka.
3. Jangan menghakimi, jangan berasumsi negatif.
4. Gunakan bahasa hangat, ramah, setara — bukan menggurui.
5. Jika ada sinyal krisis (self-harm, bunuh diri), sarankan menghubungi Into The Light Indonesia 119 ext 8.
6. Setiap respons 2–4 kalimat — fokus, tidak bertele-tele.
7. Akhiri setiap giliran dengan pertanyaan terbuka — KECUALI jika pengguna sudah berpamitan.
8. Gunakan Bahasa Indonesia natural yang cocok untuk remaja.
9. WAJIB gunakan "aku" (dirimu) dan "kamu" (pengguna). JANGAN gunakan "gue", "elo", "gw".
10. DILARANG: deskripsi tindakan fisik seperti (Mengangguk), (Tersenyum hangat), atau tanda kurung apapun.
11. DILARANG: narasi internal atau komentar di luar respons.
12. DILARANG: mengklaim memiliki perasaan atau tubuh layaknya manusia.{avoid_block}
"""


# ── Repetitive-phrase tracker ──────────────────────────────────────────────────

_VALIDATION_PHRASES: list[str] = [
    "kedengarannya", "aku mengerti", "aku paham", "aku bisa mengerti",
    "aku bisa memahami", "aku memahami", "wajar sekali", "itu wajar",
    "sangat wajar", "itu pasti", "pasti berat", "pasti sulit",
    "pasti tidak mudah", "makasih sudah berbagi", "terima kasih sudah berbagi",
    "makasih sudah mau cerita", "terima kasih sudah mau cerita",
    "aku dengar kamu", "aku ada di sini", "aku di sini",
    "kamu tidak sendirian", "kamu tidak sendiri",
    "perasaanmu valid", "perasaan kamu valid", "itu normal", "hal yang normal",
]

def _extract_used_phrases(text: str) -> set[str]:
    lower = text.lower()
    return {p for p in _VALIDATION_PHRASES if p in lower}


# ── Closing-message detector ───────────────────────────────────────────────────

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
    normalized = text.lower().strip().rstrip("!.,~")
    if normalized in _CLOSING_KEYWORDS:
        return True
    words = normalized.split()
    if len(words) <= 6:
        for kw in _CLOSING_KEYWORDS:
            if normalized.startswith(kw):
                return True
    return False


# ── Action/thinking sanitizer ──────────────────────────────────────────────────

_ACTION_PATTERN = re.compile(
    r"\(([^)]*)\)" r"|\[([^\]]*)\]" r"|\*([^*]+)\*",
    re.UNICODE,
)

def _strip_actions(text: str) -> str:
    cleaned = _ACTION_PATTERN.sub("", text)
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
        [OLLAMA_BIN, "serve"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
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
                _log(f"WARN: Model '{needed}' not listed — pull it first.")
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
    role: str
    content: str
    turn_index: int


@dataclass
class SimulatedConversation:
    dialog_id:      str
    scenario_id:    int
    kategori:       str
    topik:          str
    deskripsi:      str
    target_emotion: str
    turns:          list[Turn] = field(default_factory=list)
    metadata:       dict       = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_annotation_ready(self) -> dict:
        turns_dict = self.to_dict()["turns"]
        messages   = [{"role": t["role"], "content": t["content"]} for t in turns_dict]
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
    scenario_id = int(scenario["ID_skenario"])
    dialog_id   = f"roleplay_doh_dual_{scenario_id:04d}_{target_emotion}_{int(time.time())}"

    principles = _build_principles(scenario, agent_type)

    conv = SimulatedConversation(
        dialog_id=dialog_id,
        scenario_id=scenario_id,
        kategori=scenario["Kategori"],
        topik=scenario["Topik"],
        deskripsi=scenario["Deskripsi"],
        target_emotion=target_emotion,
        metadata={
            "user_model":       USER_MODEL,
            "chatbot_model":    CHATBOT_MODEL,
            "agent_type":       agent_type,
            "num_turn_pairs":   num_turns,
            "principles":       principles,
            "chatbot_principles": _CHATBOT_PRINCIPLES,
            "simulated_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "method":           "roleplay-doh-dual",
        },
    )

    user_sys        = make_user_system_prompt(scenario, target_emotion, agent_type, principles)
    shared_history: list[dict] = []
    used_phrases:   set[str]   = set()

    for pair_idx in range(num_turns):
        # ── USER AGENT turn — principle-adherence reminder (same as Baseline D) ─
        adherence_reminder = _make_principle_adherence_reminder(
            pair_idx, num_turns, principles, target_emotion
        )

        if pair_idx == 0:
            user_model_messages = [{"role": "user", "content": adherence_reminder}]
        else:
            user_model_messages = []
            for msg in shared_history:
                if msg["role"] == "user":
                    user_model_messages.append({"role": "assistant", "content": msg["content"]})
                else:
                    user_model_messages.append({"role": "user", "content": msg["content"]})
            user_model_messages.append({"role": "user", "content": adherence_reminder})

        user_reply = _strip_actions(ollama_chat(
            model=USER_MODEL,
            messages=user_model_messages,
            system=user_sys,
            base_url=base_url,
            temperature=0.9,
        ))
        turn_idx_u = pair_idx * 2

        if not user_reply:
            _log(f"    [EMPTY] User reply empty at turn {turn_idx_u} — closing.")
            closing_bot = "Oke, sepertinya kamu butuh waktu. Kalau mau cerita lagi, Teta selalu di sini ya."
            conv.turns.append(Turn(role="assistant", content=closing_bot, turn_index=turn_idx_u + 1))
            break

        shared_history.append({"role": "user", "content": user_reply})
        conv.turns.append(Turn(role="user", content=user_reply, turn_index=turn_idx_u))
        _log(f"    [U{turn_idx_u}] {user_reply[:100].replace(chr(10), ' ')}...")

        closing = _is_closing_message(user_reply)

        # ── CHATBOT AGENT turn — NEW: counseling principle-adherence reminder ──
        # Appended to the chatbot system prompt (not conversation history) so the
        # shared transcript stays clean and the chatbot sees it as authoritative
        # meta-guidance rather than another user message.
        chatbot_sys = make_chatbot_system_prompt(
            avoid_phrases=used_phrases,
            first_turn=(pair_idx == 0),
        )
        chatbot_reminder = _make_chatbot_adherence_reminder(pair_idx, num_turns)
        chatbot_sys_with_reminder = chatbot_sys + chatbot_reminder

        bot_reply = ollama_chat(
            model=CHATBOT_MODEL,
            messages=shared_history,
            system=chatbot_sys_with_reminder,
            base_url=base_url,
            temperature=0.75,
        )
        shared_history.append({"role": "assistant", "content": bot_reply})
        turn_idx_b = pair_idx * 2 + 1
        conv.turns.append(Turn(role="assistant", content=bot_reply, turn_index=turn_idx_b))
        _log(f"    [B{turn_idx_b}] {bot_reply[:100].replace(chr(10), ' ')}...")

        used_phrases.update(_extract_used_phrases(bot_reply))

        if closing:
            _log(f"    [CLOSING] Farewell at turn {turn_idx_u} — ending.")
            break

    return conv


# ── Output helpers ─────────────────────────────────────────────────────────────

def pair_filename(scenario_id: int, emotion: str, agent_type: str, run_ts: str) -> str:
    safe_emotion = emotion.replace(" ", "_")
    return f"roleplay_doh_dual_{scenario_id:04d}_{safe_emotion}_{agent_type}_{run_ts}.json"


def pair_is_done(output_dir: Path, scenario_id: int, emotion: str, agent_type: str, run_ts: str) -> bool:
    return (output_dir / pair_filename(scenario_id, emotion, agent_type, run_ts)).exists()


def write_pair(output_dir: Path, record: dict, agent_type: str, run_ts: str) -> Path:
    dest = output_dir / pair_filename(
        record["scenario_id"], record["target_emotion"], agent_type, run_ts
    )
    dest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


# ── Manifest loader ────────────────────────────────────────────────────────────
# Two patterns are tried in order:
#   1. Psydial / D-series format: *_{sid:04d}_{emotion}_{agent_type}_{YYYYMMDD}_{HHMMSS}.json
#   2. Legacy simulated-sft-fin format: *_{sid:04d}_{emotion}_{unix_ts}.json
_MANIFEST_RE_WITH_AGENT = re.compile(
    r".*_(\d{4})_([a-z]+)_(verbose|pendiam|menghindar|defensif)_\d{8}_\d{6}\.json$"
)
_MANIFEST_RE_NO_AGENT = re.compile(r".*_(\d{4})_([a-z_]+)_\d+\.json$")


def _load_manifest(manifest_dir: Path) -> list[tuple[int, str, str | None]]:
    """Return sorted list of (scenario_id, emotion, agent_type_or_None) triples."""
    pairs: list[tuple[int, str, str | None]] = []
    seen:  set[tuple[int, str, str | None]]  = set()
    for f in sorted(manifest_dir.glob("*.json")):
        m = _MANIFEST_RE_WITH_AGENT.match(f.name)
        if m:
            key: tuple[int, str, str | None] = (int(m.group(1)), m.group(2), m.group(3))
        else:
            m = _MANIFEST_RE_NO_AGENT.match(f.name)
            if not m:
                continue
            key = (int(m.group(1)), m.group(2), None)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return sorted(pairs)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roleplay-doh Dual Adherence: principle-adherence prompting for BOTH user and chatbot agents."
    )
    parser.add_argument("--xlsx",                  default=DEFAULT_XLSX,       help="Path to scenario Excel file")
    parser.add_argument("--output-dir",            default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--manifest",              default=None,               help="Directory of reference dialogs; derives (scenario_id, emotion) work list from filenames for fair cross-baseline comparison")
    parser.add_argument("--scenario-ids",          nargs="*", type=int,        help="Specific scenario IDs — ignored when --manifest is set")
    parser.add_argument("--emotions-per-scenario", type=int, default=1,        help="Emotions sampled per scenario (default: 1) — ignored when --manifest is set")
    parser.add_argument("--turns",                 type=int, default=None,     help=f"Turn pairs per dialogue (default: random {TURNS_MIN}–{TURNS_MAX})")
    parser.add_argument("--agent-type",            default="random",           help=f"Agent type: {USER_AGENT_TYPES + ['random']} (default: random)")
    parser.add_argument("--ollama-url",            default=DEFAULT_OLLAMA_URL, help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--resume",                action="store_true",        help="Skip already-completed pairs")
    parser.add_argument("--dry-run",               action="store_true",        help="Print plan without calling Ollama")
    args = parser.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")

    xlsx_path = Path(args.xlsx).resolve()
    if not xlsx_path.exists():
        sys.exit(f"[ERROR] Scenario file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    required_cols = {"ID_skenario", "Kategori", "Topik", "Deskripsi"}
    if not required_cols.issubset(df.columns):
        sys.exit(f"[ERROR] Missing columns. Expected {required_cols}, got {set(df.columns)}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.agent_type != "random" and args.agent_type not in USER_AGENT_TYPES:
        sys.exit(f"[ERROR] Unknown --agent-type '{args.agent_type}'. Choose from: {USER_AGENT_TYPES + ['random']}")

    # ── Build work list ────────────────────────────────────────────────────────
    # Each entry is (scenario_dict, emotion, agent_type_or_None).
    # agent_type is non-None only when derived from a manifest that encodes it
    # (e.g. psydial filenames).  None means resolve at run time via --agent-type.
    work_list: list[tuple[dict, str, str | None]] = []

    if args.manifest:
        manifest_dir = Path(args.manifest).resolve()
        if not manifest_dir.is_dir():
            sys.exit(f"[ERROR] --manifest path is not a directory: {manifest_dir}")
        manifest_pairs = _load_manifest(manifest_dir)
        has_agent = any(a is not None for _, _, a in manifest_pairs)
        _log(
            f"Manifest: {len(manifest_pairs)} unique (scenario_id, emotion"
            f"{', agent_type' if has_agent else ''}) entries from {manifest_dir}"
        )
        scenario_by_id = {int(row["ID_skenario"]): row.to_dict() for _, row in df.iterrows()}
        missing_ids: list[int] = []
        for sid, emotion, manifest_agent in manifest_pairs:
            if sid not in scenario_by_id:
                missing_ids.append(sid)
                continue
            work_list.append((scenario_by_id[sid], emotion, manifest_agent))
        if missing_ids:
            _log(f"WARN: {len(missing_ids)} scenario ID(s) from manifest not found in xlsx: {sorted(set(missing_ids))}")
    else:
        if args.scenario_ids:
            df = df[df["ID_skenario"].isin(args.scenario_ids)]
            if df.empty:
                sys.exit(f"[ERROR] No scenarios found for IDs: {args.scenario_ids}")
        for _, row in df.iterrows():
            scenario     = row.to_dict()
            kategori     = scenario["Kategori"]
            emotion_pool = CATEGORY_EMOTION_MAP.get(kategori, ALL_EMOTIONS)
            sampled      = random.sample(emotion_pool, k=min(args.emotions_per_scenario, len(emotion_pool)))
            for emotion in sampled:
                work_list.append((scenario, emotion, None))

    if args.resume:
        before      = len(work_list)
        work_list   = [
            (s, e, a) for s, e, a in work_list
            if not any(
                pair_is_done(output_dir, int(s["ID_skenario"]), e, at, run_ts)
                for at in ([a] if a is not None else
                           (USER_AGENT_TYPES if args.agent_type == "random" else [args.agent_type]))
            )
        ]
        if before - len(work_list):
            _log(f"Resume: skipping {before - len(work_list)} already-completed pair(s).")

    if args.dry_run:
        _log(f"DRY-RUN — {len(work_list)} pair(s) would run (run_ts={run_ts}):")
        for s, e, a in work_list:
            agent_type = a if a is not None else (
                random.choice(USER_AGENT_TYPES) if args.agent_type == "random" else args.agent_type
            )
            turns      = args.turns or random.randint(TURNS_MIN, TURNS_MAX)
            principles = _build_principles(s, agent_type)
            fname      = pair_filename(int(s["ID_skenario"]), e, agent_type, run_ts)
            print(f"  {fname}  turns={turns}  agent={agent_type:<12s}  principles={len(principles)}  |  {s['Topik']}")
        return

    ensure_ollama_running(args.ollama_url)
    check_models(args.ollama_url)

    _log(f"Roleplay-doh Dual Adherence simulation: {len(work_list)} pair(s)")
    _log(f"Run timestamp : {run_ts}")
    _log(f"User model    : {USER_MODEL}")
    _log(f"Chatbot model : {CHATBOT_MODEL}")
    _log(f"Agent type    : {args.agent_type}")
    _log(f"Output dir    : {output_dir}")

    written = 0
    failed  = 0
    t0      = time.time()

    for idx, (scenario, emotion, manifest_agent) in enumerate(work_list, start=1):
        sid        = int(scenario["ID_skenario"])
        num_turns  = args.turns or random.randint(TURNS_MIN, TURNS_MAX)
        agent_type = manifest_agent if manifest_agent is not None else (
            random.choice(USER_AGENT_TYPES) if args.agent_type == "random" else args.agent_type
        )
        principles = _build_principles(scenario, agent_type)
        fname      = pair_filename(sid, emotion, agent_type, run_ts)

        _log(
            f"[{idx}/{len(work_list)}] {fname}"
            f"  | {scenario['Topik']}"
            f"  | emotion={emotion}"
            f"  | turns={num_turns}"
            f"  | agent={agent_type}"
            f"  | principles={len(principles)}"
        )

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
            eta_s     = int(elapsed / idx * remaining)
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
