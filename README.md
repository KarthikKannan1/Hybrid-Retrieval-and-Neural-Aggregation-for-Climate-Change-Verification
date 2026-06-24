# Hybrid Retrieval and Neural Aggregation for Climate Change Verification

An automated fact-checking system for climate-related claims, built on the CLIMATE-FEVER task. Given a claim, the system retrieves the most relevant evidence passages from a large corpus and classifies the claim as **SUPPORTS**, **REFUTES**, **NOT_ENOUGH_INFO**, or **DISPUTED**.

The pipeline combines classical lexical retrieval (BM25) with transformer-based semantic re-ranking, then aggregates the top evidence into a claim-level verdict using a BiLSTM over frozen encoder embeddings. The best configuration reaches **53.90% dev accuracy** — an **18.84-point absolute improvement** over the task baseline of 35.06%.

Developed as a team research project for COMP90042 (Natural Language Processing), Master of Computer Science, University of Melbourne.

## System Architecture

![System architecture](docs/system-architecture.png)

The pipeline runs in two main stages:

**1. Retrieval**
- **BM25** retrieves the top-100 candidate passages for each claim.
- A **cross-encoder re-ranker** (`ms-marco-MiniLM-L-6-v2`) re-scores candidates by semantic relevance and narrows them to the top 5.
- Retrieval operates over a **curated 3,443-passage pool** distilled from the full 1.2M-passage corpus, which substantially improved retrieval quality (F-score 0.10 → 0.25).

**2. Aggregation & classification** — two strategies compared:
- **NLI consistency aggregator** — an NLI cross-encoder labels each sentence (support / refute / neutral), with a rule-based consistency check that explicitly targets the hard DISPUTED class.
- **Neural aggregator (best)** — frozen encoder embeddings (BERT, RoBERTa, or DeBERTa-v3) summarised by a **BiLSTM**, then a feed-forward head producing the four-way label. Trained with inverse-frequency class weights to counter class imbalance.

## Key Results

| Configuration | Dev Accuracy |
|---|---|
| Baseline (spec sheet) | 35.06% |
| NLI consistency aggregator only | 27.92% |
| BiLSTM (BERT-base) | 48.05% |
| BiLSTM (DeBERTa-v3-base) | 51.30% |
| **BiLSTM (RoBERTa-base)** | **53.90%** |

Findings in brief: the BiLSTM aggregator consistently beat a unidirectional LSTM, the choice of frozen encoder mattered more than the aggregator design (RoBERTa transferred best to the climate domain), and the DISPUTED class remained unsolved (F1 = 0.00) — consistent with it being flagged as open future work in the original CLIMATE-FEVER paper. Full analysis, training curves, and confusion matrix are in the report.

## Method Highlights

- **Hybrid retrieval** — pairs BM25's lexical matching with a cross-encoder's semantic scoring, rather than relying on either alone.
- **Curated evidence pool** — filtering 1.2M passages down to a climate-specific pool was the single biggest retrieval gain.
- **Frozen encoders** — embeddings are pre-computed once and cached, which removes the encoder from the training loop and prevents a large model from overfitting a small (~1.2k claim) training set.
- **Sentence-level NLI reasoning** — an explicit per-sentence verdict stage aimed at the conflicting-evidence (DISPUTED) case that prior pipelines couldn't handle.

## Running the Code

The system is implemented as a single Jupyter notebook (`FinalNotebook.ipynb`), designed to run on Google Colab.

1. Open the notebook in Google Colab.
2. Set the runtime to **GPU** (Runtime → Change runtime type → GPU). This matters a lot: the transformer encoding steps run in roughly half an hour on a GPU but many hours on CPU.
3. Provide the CLIMATE-FEVER data files (`train-claims.json`, `dev-claims.json`, `test-claims-unlabelled.json`, `evidence.json`) — these are not included in the repo.
4. Run all cells top to bottom.

The notebook is kept intact as a single document so that the full research narrative — code, training logs, plots, and analysis — is visible inline without re-running.

## Report

The full research report (ACL format), including literature review, methodology, experiments, and analysis, is included: see [Project Report](report.pdf).

## Team & Contributions

A three-person project (Group 107):

- **Karthik Kannan** — literature review, system design, and the retrieval pipeline (BM25 + cross-encoder re-ranking, curated evidence pool).
- **Ivan Smilenov** — encoder and classifier implementation, running experiments, finalizing structure.
- **Ardelia Shaula Araminta** — data analysis and report writing.

## License & Use

This is academic coursework completed as a team. Please don't reuse it as a basis for COMP90042 submissions. If you build on the ideas, a citation or acknowledgement is appreciated.
