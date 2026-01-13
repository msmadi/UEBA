# UEBA
Multi-Axis Trust Modeling for Interpretable Account Hijacking Detection

# Hadith-Inspired Trust Modeling for UEBA

This repository contains the reference implementation and experimental
artifacts for the paper:

**“Multi-Axis Trust Modeling for Interpretable Account Hijacking Detection”**

The code implements a structured, interpretable feature framework for
detecting account hijacking and insider threats in user activity logs,
together with temporal extensions for sequence-aware detection.

---

## Overview

User and Entity Behavior Analytics (UEBA) systems often rely on opaque models
and low-level count features. This project introduces a **Hadith-inspired
multi-axis trust model**, translating classical trust criteria into
behavioral features grouped along five axes:

- **ʿAdālah (Integrity / Long-Term Stability)**
- **Ḍabṭ (Precision / Hygiene)**
- **Isnād (Contextual / Network Continuity)**
- **Reputation (Jarḥ wa-Taʿdīl)**
- **Anomaly Evidence (Shudhūdh / ʿIllah)**

We further extend this representation with **temporal sequence features**
that capture short-horizon behavioral drift across consecutive windows.

The framework is evaluated on:
- **CLUE-LDS** (cloud activity logs with injected hijack scenarios)
- **CERT Insider Threat Dataset r6.2** (realistic insider threat benchmark)

---

## Repository Structure


