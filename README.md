# PII Privacy Intelligence Platform

A production-oriented **hybrid PII extraction and privacy intelligence platform** for identifying Personally Identifiable Information from structured and unstructured documents using OCR, deterministic extraction, fine-tuned transformer-based NER, contextual reasoning, and agentic orchestration.

The system is designed for Indian identity, employee, customer, KYC, healthcare, education, employment, travel, insurance, and application documents, including noisy OCR-derived text.

---

## Overview

The PII Privacy Intelligence Platform combines multiple extraction strategies instead of relying on a single model.

The current architecture uses:

* **PaddleOCR** with Tesseract fallback for scanned documents
* **Regex and rule-based extraction** for highly structured PII
* **Fine-tuned DeBERTa-based token classification** for contextual PII extraction
* **Context-aware entity classification** to distinguish primary-person information from unrelated or secondary-person information
* **Local LLM fallback using Ollama** for ambiguous or conflicting cases
* **LangGraph-based orchestration** for coordinating extraction, contextual reasoning, reconciliation, and fallback
* **Entity validation, normalization, reconciliation, and confidence scoring**
* **FastAPI** backend for document processing and analytics
* **React + Vite** frontend for interactive PII analysis
* Reproducible evaluation and training pipelines

The architecture is intentionally modular so that deterministic extraction, machine-learning models, OCR, contextual reasoning, and future models can be improved independently.

---

## Key Capabilities

### Multi-format document processing

The platform is designed to process:

* PDF documents
* Scanned PDFs
* DOCX documents
* XLS/XLSX spreadsheets
* PNG images
* JPG/JPEG images
* OCR-derived document text

### PII categories

The current extraction pipeline supports:

* NAME
* ADDRESS
* DOB
* PHONE
* EMAIL
* PAN
* AADHAAR
* DRIVING_LICENCE
* VOTER_ID

The architecture allows additional PII categories to be introduced without redesigning the complete pipeline.

---

# System Architecture

```text
                    ┌─────────────────────────┐
                    │      User / Client      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     React Frontend      │
                    │      Vite + React       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       FastAPI API        │
                    │     Document Upload      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Document Ingestion    │
                    │ PDF / DOCX / XLSX / Img │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              Text available             OCR required
                    │                         │
                    │                         ▼
                    │              ┌────────────────────┐
                    │              │     PaddleOCR      │
                    │              │         │          │
                    │              │         ▼          │
                    │              │     Tesseract      │
                    │              │     fallback       │
                    │              └─────────┬──────────┘
                    │                        │
                    └────────────┬───────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Extraction Pipeline   │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
       ┌──────────────┐  ┌──────────────┐  ┌────────────────┐
       │ Regex / Rules│  │   DeBERTa    │  │    Context     │
       │ Structured   │  │ Token Class. │  │ Classification │
       │ PII          │  │ Semantic PII │  │                │
       └──────┬───────┘  └──────┬───────┘  └───────┬────────┘
              │                 │                  │
              └─────────────────┼──────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │ Candidate Reconciliation│
                    │ Normalization           │
                    │ Validation              │
                    │ Deduplication           │
                    │ Conflict Resolution     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Ambiguity / Conflict    │
                    │ Detection               │
                    └────────────┬────────────┘
                                 │
                          Required only
                          when ambiguity exists
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Local LLM Fallback    │
                    │       Ollama             │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Final Entity Resolution │
                    │ + Confidence Scoring     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ PII Results + Analytics │
                    └─────────────────────────┘
```

---

# Extraction Strategy

The platform follows a **hybrid extraction architecture**.

### 1. Deterministic extraction

Regex-based extraction handles strongly structured identifiers where deterministic patterns provide high precision.

This layer is responsible primarily for:

* PAN
* Aadhaar
* Phone numbers
* Email addresses
* Driving licence numbers
* Voter IDs
* Structured date patterns

This reduces unnecessary model inference and provides reliable extraction for strongly formatted identifiers.

### 2. Transformer-based semantic extraction

A fine-tuned **DeBERTa token-classification model** handles contextual entities where fixed patterns are insufficient.

The primary semantic targets are:

* NAME
* ADDRESS
* DOB

The transformer operates over document context rather than treating each candidate independently.

### 3. Contextual entity reasoning

Extracted candidates are evaluated against their surrounding document context.

This allows the pipeline to distinguish between:

* Primary person's information
* Secondary-person information
* Organization information
* Non-target dates
* Non-target addresses
* Contextually unrelated contact information

### 4. LLM fallback

A local Ollama-hosted LLM is invoked only when the deterministic and transformer pipeline identifies ambiguity, conflicts, or insufficient contextual confidence.

This keeps the expensive contextual reasoning path selective instead of making an LLM call for every document.

### 5. Reconciliation

Candidates generated by different extraction methods are normalized and reconciled before producing the final result.

The reconciliation layer handles:

* Entity normalization
* Validation
* Duplicate removal
* Candidate comparison
* Source prioritization
* Overlapping entities
* Cross-method agreement
* Contextual filtering
* Final candidate selection

---

# Agentic Orchestration

The extraction pipeline uses **LangGraph** to orchestrate the individual processing stages.

The graph coordinates:

1. Regex extraction
2. Transformer-based extraction
3. Context classification
4. Preliminary reconciliation
5. Ambiguity detection
6. Conditional LLM fallback
7. Final reconciliation

This provides a structured execution graph rather than a monolithic extraction function.

The architecture also leaves room for additional specialized agents for:

* Validation
* Extraction
* Reconciliation
* Contextual reasoning
* Document-type routing
* Model selection

---

# Confidence and Validation

Every final entity is enriched with metadata including:

* Extraction source
* Confidence score
* Confidence level
* Format validation
* Method agreement
* Contextual role
* Entity type
* Normalized value

Confidence levels are grouped into:

* **High:** ≥ 0.90
* **Medium:** ≥ 0.70 and < 0.90
* **Low:** < 0.70

This enables downstream applications to distinguish highly reliable PII from entities requiring additional review.

---

# Current Benchmark Performance

The current stable pipeline was evaluated against a **frozen 150-document benchmark** covering multiple Indian PII document categories.

### Overall performance

| Metric              |     Result |
| ------------------- | ---------: |
| Documents evaluated |    **150** |
| True Positives      |    **606** |
| False Positives     |     **91** |
| False Negatives     |     **39** |
| Precision           | **86.94%** |
| Recall              | **93.95%** |
| F1 Score            | **90.31%** |

### Entity-level performance

| Entity Type     | Precision |  Recall |          F1 |
| --------------- | --------: | ------: | ----------: |
| AADHAAR         |   100.00% |  93.88% |  **96.84%** |
| ADDRESS         |    86.67% |  91.92% |  **89.22%** |
| DOB             |    78.53% | 100.00% |  **87.98%** |
| DRIVING_LICENCE |   100.00% | 100.00% | **100.00%** |
| EMAIL           |    87.72% | 100.00% |  **93.46%** |
| NAME            |    81.33% |  81.33% |  **81.33%** |
| PAN             |   100.00% | 100.00% | **100.00%** |
| PHONE           |    98.04% | 100.00% |  **99.01%** |
| VOTER_ID        |   100.00% | 100.00% | **100.00%** |

---

# Document-Level Performance

| Document Type         | Precision |  Recall |          F1 |
| --------------------- | --------: | ------: | ----------: |
| Aadhaar               |    88.35% |  91.00% |  **89.66%** |
| Bank KYC Form         |    90.62% |  93.55% |  **92.06%** |
| Customer Registration |    93.33% |  96.55% |  **94.92%** |
| Driving Licence       |    52.10% |  79.49% |  **62.94%** |
| Employee Information  |   100.00% | 100.00% | **100.00%** |
| Hospital Registration |    84.38% |  93.10% |  **88.52%** |
| Insurance Form        |    96.67% |  96.67% |  **96.67%** |
| Job Application       |    93.55% |  96.67% |  **95.08%** |
| PAN                   |   100.00% | 100.00% | **100.00%** |
| Rental Tenant Form    |   100.00% | 100.00% | **100.00%** |
| School / College Form |    93.33% |  96.55% |  **94.92%** |
| Service Customer Form |    86.67% |  89.66% |  **88.14%** |
| Travel Passenger Form |   100.00% | 100.00% | **100.00%** |
| Voter ID              |    96.88% |  96.88% |  **96.88%** |

---

# Model and Pipeline Statistics

The current benchmark extraction distribution demonstrates the hybrid architecture:

| Source        | Entities |     Share |
| ------------- | -------: | --------: |
| DeBERTa       |      544 | **77.5%** |
| Regex         |      149 | **21.2%** |
| LLM / Context |        9 |  **1.3%** |

The LLM is intentionally used as a **selective fallback** rather than the primary extraction engine.

### LLM fallback

| Metric                      |    Result |
| --------------------------- | --------: |
| Documents evaluated         |       150 |
| Documents invoking fallback |         8 |
| Fallback rate               | **5.33%** |

This demonstrates that most documents can be resolved without invoking the slower contextual reasoning path.

---

# Latency

Measured on the current local development environment:

| Metric  |                 Result |
| ------- | ---------------------: |
| Average | **1.604 s / document** |
| Median  | **33.1 ms / document** |
| P95     | **6.579 s / document** |

The latency distribution is intentionally reported using both median and tail latency because the conditional OCR and LLM paths create significantly different execution costs across documents.

---

# Training Pipeline

The repository contains a reproducible transformer training pipeline supporting:

* Dataset preparation
* Entity label mapping
* BIO tagging
* Tokenization
* Transformer fine-tuning
* Validation
* Local inference

The current training workflow is designed around synthetic Indian-document PII data and will be extended with realistic OCR corruption to improve robustness against noisy document extraction.

The next model iteration specifically targets improvements in:

* NAME boundary detection
* ADDRESS extraction
* DOB contextual disambiguation
* OCR-corrupted PII recognition

The benchmark used for final evaluation remains frozen to prevent iterative tuning against the evaluation set.

---

# Technology Stack

### Backend

* Python
* FastAPI
* LangGraph
* PyTorch
* Hugging Face Transformers
* DeBERTa
* PaddleOCR
* Tesseract OCR
* PyMuPDF
* python-docx
* pandas
* OpenPyXL
* RapidFuzz
* Requests

### AI / ML

* Fine-tuned DeBERTa token classification
* Rule-based PII extraction
* Context-aware entity classification
* Local LLM inference through Ollama
* Confidence-based candidate selection
* Hybrid model orchestration

### Frontend

* React
* Vite
* JavaScript
* CSS

### Development

* Conda
* Git
* Docker-ready architecture
* Local model inference

---

# Project Structure

```text
PII_Privacy_Platform/
│
├── backend/
│   ├── agents/
│   │   ├── context_agent.py
│   │   └── graph.py
│   │
│   ├── api/
│   │   └── routes.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   │
│   ├── evaluation/
│   │   ├── dataset/
│   │   └── training/
│   │
│   ├── extraction/
│   │   ├── contextual.py
│   │   ├── nlp.py
│   │   ├── normalize.py
│   │   └── regex.py
│   │
│   ├── ingestion/
│   │   ├── docx.py
│   │   ├── excel.py
│   │   ├── loader.py
│   │   └── pdf.py
│   │
│   ├── models/
│   │   ├── context_classifier.py
│   │   └── schemas.py
│   │
│   ├── ocr/
│   │   └── engine.py
│   │
│   ├── services/
│   │   ├── candidate_scoring.py
│   │   ├── confidence.py
│   │   ├── extraction_service.py
│   │   └── reconciliation.py
│   │
│   ├── transformer/
│   │   ├── dataset.py
│   │   ├── inference.py
│   │   ├── labels.py
│   │   ├── prepare_data.py
│   │   ├── tokenize.py
│   │   └── train.py
│   │
│   ├── tests/
│   ├── app.py
│   ├── Dockerfile
│   ├── Modelfile
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/
    └── React + Vite application
```

---

# API

The backend exposes FastAPI endpoints for:

* Health monitoring
* PII extraction
* Batch document processing
* Extraction analytics

The API is structured to support future deployment behind a reverse proxy or application gateway without requiring architectural changes to the extraction engine.

---

# Design Principles

### Hybrid over single-model extraction

Structured PII is handled deterministically where possible, while contextual information is delegated to the transformer and reasoning layers.

### Model-first semantic extraction

Semantic PII extraction is driven primarily by a fine-tuned transformer rather than relying exclusively on handcrafted contextual rules.

### Selective LLM usage

The LLM is used as a fallback for ambiguity and conflict resolution rather than as the default extraction mechanism.

### Reproducible evaluation

A frozen benchmark is maintained separately from training data to provide a stable measure of model improvements.

### Modular architecture

OCR, ingestion, extraction, orchestration, model inference, reconciliation, validation, and analytics are isolated into separate components.

### Local-first privacy

The current system is designed around local processing and local model inference, minimizing the need to transmit document contents to external AI APIs.

---

# Current Development Status

The stable baseline currently achieves:

**90.31% entity-level F1**

with:

* **93.95% recall**
* **86.94% precision**
* **100% F1** for PAN, Driving Licence, and Voter ID on the current benchmark
* **96.84% F1** for Aadhaar
* **99.01% F1** for Phone
* **93.46% F1** for Email

The next model iteration focuses on improving the weaker semantic categories—particularly **NAME, ADDRESS, and DOB**—through additional DeBERTa fine-tuning on OCR-realistic synthetic document data.

---

# Future Direction

Planned improvements include:

* OCR-aware transformer fine-tuning
* Improved NAME boundary detection
* Improved ADDRESS span detection
* Better contextual DOB classification
* More realistic synthetic document generation
* Additional document layouts
* Model confidence calibration
* Document-type-aware routing
* Larger-scale evaluation
* GPU-assisted inference
* Containerized deployment
* Production monitoring and analytics
* Optional external model serving
* Scalable persistence when deployment requirements justify it

The architecture is intentionally designed to evolve from a local-first system into a production deployment without requiring a complete rewrite of the extraction pipeline.
