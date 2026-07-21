# agent-simulation-mental-id

Data and code repository for the conference paper **"Evaluating Indonesian Mental Health Chatbots via Agentic Simulation: When Ethics and Realism Win."**

> ⚠️ **Content warning**: this repository contains data and examples that may be sensitive or upsetting, including references to self-harm, distress, and mental health crises, written in Bahasa Indonesia.

## Overview

Evaluating mental health chatbots typically relies on costly, slow human studies: psychologists and target users must iteratively validate data construction, rate conversation trials, and rank model preferences before any fine-tuning stage can be trusted. This project proposes an **agentic simulation framework** that replaces that bottleneck with two LLM agents that role-play a multi-turn counseling conversation, which is then scored by an ensemble of LLM judges — enabling reproducible, low-cost evaluation of Indonesian adolescent mental health chatbots.

A common assumption in chatbot design is that framing the chatbot as a professional, human-like counselor ("anthropomorphizing" it) produces richer, more therapeutic conversation. This work tests that assumption directly and finds the opposite: explicitly framing the chatbot agent **as an AI** (rather than as a human counselor) improves the quality of both student and chatbot responses, while better upholding the ethical principle of AI transparency.

## Framework

The simulation pipeline instantiates two LLM agents (Fig. 1 in the paper):

1. **User agent** (`gemma4:e4b`) — plays a 16–17 year-old Indonesian high school student ("help seeker"), conditioned on a predefined emotional state, scenario category, and communication persona (`verbose`, `pendiam`, `menghindar`, or `defensif`).
2. **Chatbot agent** (`gemma3:12b`, at various fine-tuning stages) — plays the mental health chatbot ("supporter").

The two agents converse for several turns; the resulting transcripts are passed through a pairwise extractor and scored by three **LLM-as-judge** models — OpenAI GPT-5.4, Anthropic Claude Sonnet-4.6, and open-sourced Qwen-3.6:27b — against independent rubrics for the user agent (5 dimensions) and the chatbot agent (9 dimensions).

We introduce and evaluate **EmoStyle**, a simple simulation approach steered only by (a) emotion and (b) persona/communication style, and compare it against three baselines adapted from English-domain agentic simulation work:

| Approach | Description | Code |
|---|---|---|
| **EmoStyle (ours)** | Simulation steered by emotion + persona/style, minimal scaffolding | [`EmoStyle/`](EmoStyle) |
| **SimPsyDial** | Adds Big-Five traits (openness, conscientiousness, extraversion, agreeableness, neuroticism) and 3 resistance levels for the user agent's willingness to share | [`SimPsyDial/`](SimPsyDial) |
| **PsyDial** | Auto-generates student role cards (gender, grade, trait personality, resistance level); masks the role card from the chatbot except trait personality and resistance level | [`PsyDial/`](PsyDial) |
| **Roleplay-doh** | Principle-adherence prompting for both agents, with principles that shift across the beginning / middle / end of the conversation | [`RolePlayDoh/`](RolePlayDoh) |

Each approach is run under two conditions: **AI-aware** (the chatbot is explicitly told it is an AI, not human) and **anthropomorphized** (the chatbot is told to act as a human counselor). See Tables I–II in the paper for the exact system prompts used in Bahasa Indonesia.

## Repository structure

```
.
├── EmoStyle/            # Our simulation method: emotion + persona-steered agent-to-agent conversation
├── SimPsyDial/           # Baseline: Big-Five traits + resistance-level steered simulation
├── PsyDial/              # Baseline: role-card-generated, partially masked student simulation
├── RolePlayDoh/          # Baseline: principle-adherence prompting for both agents
├── data/
│   ├── finetuning/       # Fine-tuning data (e.g. ESConv translated/augmented) for the chatbot agent
│   ├── simulation/       # Contextual data used to condition the simulation:
│   │                     #   emotion list, role cards, resistance levels, Big-Five traits,
│   │                     #   communication styles, and principle sets (EmoStyle / Roleplay-doh)
│   └── evaluation/       # Evaluation rubrics and lexicons used to assess simulated conversations:
│                         #   student & chatbot scoring rubrics, emotion lexicon, harmful-content lexicon
├── paper/                # Anonymized conference paper (PDF)
└── LICENSE
```

## Dataset

Data used at each stage of the study (paper Table III):

| Stage | Data | Open-sourced | #Samples | Avg. turns |
|---|---|:---:|---:|---:|
| Fine-tuning | ESConv (translated & augmented) | ✔ | 157 | 21 |
| Fine-tuning | Interview data (ours) | ✘ | 18 | 65 |
| Fine-tuning | CALM-ID (ours) | ✘ | 17,989 | 30 |
| Preference tuning | Preference data (ours) | ✘ | 528 | 19 |
| Simulation | Emotion data (adopted) | ✔ | 41 | N/A |
| Simulation | Scenario (ours) | ✘ | 147 | N/A |
| Simulation | Role cards (adopted from PsyDial) | ✔ | 8 | N/A |
| Simulation | User traits (adopted from PsyDial) | ✔ | 15 | N/A |
| Simulation | Resistance level (adopted from PsyDial) | ✔ | 3 | N/A |
| Simulation | User agent style (ours) | ✔ | 4 | N/A |
| Evaluation | Student rubrics | ✔ | 5 | N/A |
| Evaluation | Chatbot rubrics | ✔ | 9 | N/A |
| Evaluation | Harsh insults / negative colloquials | ✔ | 101 | N/A |

## Chatbot agents and LLM judges

We evaluate `gemma3:12b` across different training/fine-tuning stages: (i) base instruction model, no fine-tuning; (ii) SFT70 — supervised fine-tuning stopped early at 70% training progress; (iii) SFT — supervised fine-tuning at 100% progress; (iv) DPO — preference-tuned via Direct Preference Optimization; and (v) `gemma4:e4b` as a more recent base model, used without fine-tuning on our dataset.

Three LLM-as-judge families score every conversation: **OpenAI GPT-5.4**, **Anthropic Claude Sonnet-4.6**, and **Ollama Qwen-3.6:27b** (open-sourced).

## Evaluation rubrics

**Student (user) agent — 5 dimensions**, following prior agentic-simulation work:
Authenticity (AU), Stayed-in-Role (SIR), Mirroring-Real-World-Challenges (RW), Ready-as-Training-Partner (RTP), Recommend-to-novices (RN).

**Chatbot agent — 9 dimensions**, following prior dialogue-evaluation work:
Coherence (CO), Empathy (EM), Problem-Understanding (PU), Intervention (IN), Emotion-Improvement (EI), Safety (SA), Language-and-Cultural (LC), Engagement (EN), Non-Judgmental (NJ).

See [`data/evaluation/student_rubrics.md`](data/evaluation/student_rubrics.md) and [`data/evaluation/chatbot_rubrics.md`](data/evaluation/chatbot_rubrics.md) for the full scoring scales.

## Key findings

- **EmoStyle outperforms baselines.** A simple simulation approach steered only by emotion and persona/style outperforms SimPsyDial, PsyDial, and Roleplay-doh across most rubric dimensions for both the student and chatbot agents.
- **AI-aware framing wins.** Across nearly every rubric dimension, explicitly framing the chatbot as an AI (rather than as a human counselor) produces *more* authentic, human-like conversational data than anthropomorphized framing — challenging the common assumption that anthropomorphism is necessary for empathetic responses.
- **Anthropomorphized chatbots sustain the human-counselor persona even when it may not be appropriate**, whereas AI-aware chatbots consistently disclose their AI nature while still redirecting the conversation back to a therapeutic thread (see Table VI in the paper for a side-by-side example).
- **Fine-tuning stage matters.** Win/loss tournaments across fine-tuning stages (base, SFT70, SFT, DPO) show `gemma4:e4b` outperforming earlier-stage `gemma3:12b` variants overall, with a narrower gap on Emotion-Improvement, Safety, Language-and-Cultural, and Non-Judgmental dimensions — plausibly because all evaluated models share the same `gemma` family and inherited safety alignment.

## Limitations

This work is a preliminary investigation into the effects of anthropomorphizing a chatbot agent, constrained by the scope of available datasets and resources. The authors note that further research is needed to extend these experiments and to incorporate human stakeholders — targeted users and mental health professionals — directly into the evaluation process.

## License

This repository is released under the [Apache License 2.0](LICENSE).
