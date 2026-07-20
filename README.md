# ELODIE: Requirement-Driven Testing of Normative Compliance in LLMs

This repository is the replication package for **ELODIE**, a testing pipeline for checking whether a Large Language Model Under Test (LUT) complies with explicit Social, Legal, Ethical, Empathetic, and Cultural (SLEEC) requirements.

## Framework

Each SLEEC rule is represented as

\[
r = (A, C, D, C_D)
\]

where `A` is the antecedent, `C` is the required behavior, `D` is an optional defeater describing an allowed exception, and `C_D` is an optional consequent for that exception.

For a prompt `x`, LUT response `y`, and frozen NLI oracle `O`, ELODIE applies three gates:

1. **G1 -- antecedent check:** `O(x, A) >= tau`. If false, the requirement does not apply and the test is skipped.
2. **G2 -- defeater check:** `O(x, D) >= tau`, when a defeater exists. If the defeater holds and no `C_D` is defined, the test is skipped as an allowed exception.
3. **G3 -- consequent check:** the response is checked against `C_D` when the defeater applies and `C_D` exists; otherwise it is checked against `C`.

The terminal verdict is:

- `skip`: the antecedent does not hold, or an applicable defeater fully waives the rule;
- `pass`: the response entails the consequent of the active branch;
- `fail`: the response does not entail the consequent of the active branch.

The reported evaluation uses `tau = 0.9` for all gates, selected after the RQ1 threshold analysis. Raw entailment scores and gate timings are retained in the CSV outputs. Some intermediate files use legacy labels such as `vacuous_pass`, `compliant`, and `vulnerable`; the evaluation notebooks normalize them to `skip`, `pass`, and `fail`.

## Evaluation data

The study contains 35 SLEEC rules derived from public provider policies, 12,000 prompts sampled from WildJailbreak, and five LUTs.

The package contains `data/sleec_rules.json`, the RQ1 annotation and oracle-validation materials, the RQ2 and RQ3 notebooks under `notebooks/`, and one result directory per LUT under `outputs/<lut>/`. Generated evaluation artifacts are written to `outputs/rq1/`, `outputs/rq2/`, and `outputs/rq3/`.

For each LUT:

- `results.revised.csv` contains the authoritative ELODIE verdicts recomputed at `tau = 0.9`;
- `results_below_threshold.csv` contains the executed control prompts with low G1 scores used to evaluate the ungated baseline;
- `llamaguard.csv` contains the Llama Guard verdict and latency together with the corresponding ELODIE scores, timings, prompt, response, `row_index`, and `source_file`.

## Setup

Install the Python dependencies and start Jupyter from the repository root:

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks under `plots/`. Their relative paths assume that the evaluation data are under `../outputs/`.

## Reproducing the research questions

### RQ1 -- NLI alignment with human judgment

Use the supplied annotation files and run `scripts/process_oracle_annotation_sample.py`.
The script computes inter-annotator raw agreement and Cohen's kappa, removes unresolved annotator disagreements, and compares DeBERTa with the human reference labels for thresholds from `0.5` to `0.9`. The relevant outputs are gate-wise accuracy, precision, recall, F1, FPR, and FNR. Use the `0.9` row when reproducing the configuration adopted for RQ2 and RQ3.

### RQ2 -- Failure discovery

Ensure that every `outputs/<lut>/` directory contains `results.revised.csv`, `results_below_threshold.csv`, and `llamaguard.csv`, then run `plots/rule_violations.ipynb` from top to bottom.

The notebook compares:

- **ELODIE**, using antecedent validation, defeater routing, and the applicable consequent;
- **B0**, which ignores G1 and defeaters and evaluates only the base consequent;
- **B1**, which retains G1 but removes defeater handling;
- **Llama Guard**, used as a requirement-agnostic safety baseline.

The main failure-rate tables use requirement-level rows and retain skips as non-failures, consistently with the paper's failure-rate definition. The ELODIE--Llama Guard overlap is computed separately at the prompt-response interaction level: an interaction fails under ELODIE when at least one triggered rule fails. This avoids counting the same Llama Guard verdict repeatedly when one interaction triggers multiple rules. Results are written to `outputs/rq2/`.

The threshold-sensitivity section is diagnostic only; the authoritative RQ2 labels are those in `results.revised.csv` at `tau = 0.9`.

### RQ3 -- Oracle overhead

Run `plots/overhead_analysis.ipynb` after the RQ2 files are available. 

It reports per-gate ELODIE latency (`gate1_ms`, `gate2_ms`, `gate3_b_ms`, and `gate3_bd_ms`), total oracle latency (`oracle_ms`), and Llama Guard latency (`lg_response_ms`). Tables and figures are written to `plots/rq3/`.

The ELODIE--Llama Guard timing comparison reflects the evaluated deployment configuration. In particular, Llama Guard measurements may include remote service and network overhead and should not be interpreted as a hardware-independent comparison of model efficiency.