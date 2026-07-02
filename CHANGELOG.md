# Changelog

## Full pipeline — SUT clients, orchestrator, config

Turned the evaluation library into a runnable end-to-end framework.

### `src/generator/` — adapted from another project, stripped to LLM clients
- Removed all toxicity/pattern-mining machinery: tree-adherence scoring (stanza/nltk), PyTorch Lightning training, T5 paraphrasing, topic/pattern/few-shot logic, and the Watson/T5/Baseline backends.
- Reduced to a minimal `LLMClient` interface (`generate(prompt, system=?, temperature=?) -> str`) with three backends: `GeminiClient`, `DeepSeekClient` (both via a shared `OpenAICompatibleClient`), and `LocalLLMClient` (vLLM/llama.cpp HTTP). `build_llm(config)` factory.
- These clients are the **System Under Test** (Phase 3) — the piece the pipeline was missing.

### Orchestrator (Phase 3 + wiring)
- **`elodie/pipeline.py`** — `run_pipeline(oracle, rules, generate_fn, sut, thresholds, output_dir)` chains generate → SUT → gate-scoring for every rule (skipping `excluded_from_generation` rules), computes per-rule and overall EVR/RVR/MAS, and saves everything. Dependency-injected, so `elodie` stays decoupled from `generator`.
- **`src/run_pipeline.py`** — CLI wiring config → generator + SUT + oracle.

### Split thresholds + config file
- **`Thresholds(exploit, defeater, compliance)`** — the schema's independent $\tau_1/\tau_2/\tau_3$, threaded through `evaluate_case`; `CaseResult` now records all three (`t_exploit`, `t_defeater`, `t_compliance`).
- **`Oracle` no longer holds a threshold** — it is a pure scorer; thresholds live in config/evaluation.
- **`config/elodie.yaml`** + **`elodie/config.py`** (`load_config`) — one parameters file for thresholds, oracle model, generator, and SUT.

### Data registration
- Per-rule `outputs/cases/<RULE_ID>.json` (every prompt, response, gate score, threshold, verdict) plus `outputs/summary.json` (run metadata + per-rule/overall metrics).

### Also
- `requirements.txt` += `openai`, `requests`, `pyyaml`.
- `deberta_example.py` — oracle constructed without a threshold arg.
- README §7 (Implementation) added; generator README rewritten.

---

## Merge — oracle framework + Phase 1

Merged the DeBERTa oracle / gate-evaluation framework into the same `elodie` package as the Phase 1 prompt generator, so the whole pipeline shares one rule type and one namespace.

### Consolidated into `src/elodie/`
- **`oracle.py`** -- moved from the standalone `src/oracle/` package. `Oracle` (abstract base, one `score(premise, hypothesis)` method) + `DebertaOracle` (frozen DeBERTa-v3 NLI cross-encoder). Model constant renamed to `DEFAULT_DEBERTA_MODEL` to avoid colliding with `prompt_generation.DEFAULT_MODEL` (Gemini).
- **`evaluation.py`** -- moved from `src/oracle/case.py`. `CaseResult`, `Verdict`, `evaluate_case`, `save_cases`, and the EVR/RVR/MAS metrics.
  - **Key integration:** dropped the local `Rule` dataclass; `evaluate_case` now consumes the coworker's `SleecRule` directly, so a prompt is scored against the same rule it was generated from.
  - **Defeater consequents:** `Defeater.consequent` is now optional. When several defeaters exist, the strongest-entailed one over the threshold is the *active* exception. A consequent-less active defeater fully waives the rule (`SAFE_BYPASS`, `compliance = None`); an active defeater *with* a consequent `B_D` redirects the Gate 3 check to `R => B_D` instead of `R => B`. `CaseResult.compliance` is now `float | None`, and `mean_adherence_score` excludes bypassed (unscored) cases. README §3 and Phases 4/5 updated to match; Phase 5 renumbered to NLI Gate 3 for consistency with the schema.
- **`__init__.py`** -- now exports the rules, oracle, and evaluation API. `prompt_generation` is intentionally *not* imported here so the oracle side works without langchain installed; the CLI keeps importing it via `elodie.prompt_generation`.

### Removed
- `src/oracle/` package (contents merged into `src/elodie/`).

### Changed
- `src/deberta_example.py` -- imports `DebertaOracle` from `elodie` instead of `oracle`.
- `requirements.txt` -- added `transformers` and `torch` for the oracle.

---

## Added

Changes made while building the ELODIE Phase 1 pipeline (initial prompt generation). All items below are new, untracked files -- nothing pre-existing was modified.

### `data/sleec_rules.json`
One-time, hand-authored SLEEC rule set extracted from Google's [Generative AI Prohibited Use Policy](https://policies.google.com/terms/generative-ai/use-policy) (fetched from the raw page HTML for verbatim accuracy, not a paraphrased summary).

- **20 rules**, one per bullet point across the policy's four "Prohibited Uses" categories:
  - `DANGEROUS_ILLEGAL` (8): `CSAM_EXPLOITATION`, `VIOLENT_EXTREMISM`, `NON_CONSENSUAL_INTIMATE_IMAGERY`, `SELF_HARM`, `ILLEGAL_ACTIVITIES_SUBSTANCES`, `PRIVACY_IP_RIGHTS`, `UNAUTHORIZED_TRACKING`, `HIGH_RISK_AUTOMATED_DECISION`
  - `SECURITY_COMPROMISE` (3): `SPAM_PHISHING_MALWARE`, `INFRASTRUCTURE_ABUSE`, `CIRCUMVENT_SAFETY_FILTERS`
  - `HATEFUL_HARMFUL_CONTENT` (4): `HATE_SPEECH`, `HARASSMENT_BULLYING`, `VIOLENCE_INCITEMENT`, `SEXUALLY_EXPLICIT_CONTENT`
  - `MISINFORMATION_MISREPRESENTATION` (5): `FRAUD_SCAMS`, `IMPERSONATION`, `MISLEADING_EXPERTISE`, `MISLEADING_GOV_HEALTH_CLAIMS`, `MISREPRESENT_AI_PROVENANCE`
- Each rule has `antecedent`, `consequent`, and zero or more `{defeater, consequent}` pairs (19/20 rules carry at least one defeater).
- 3 rules (`CSAM_EXPLOITATION`, `NON_CONSENSUAL_INTIMATE_IMAGERY`, `SEXUALLY_EXPLICIT_CONTENT`) are flagged `excluded_from_generation: true` with an `exclusion_reason` -- fully defined as rules, but skipped by the automated prompt generator.
- Top-level metadata: `source`, `source_url`, `source_last_modified`, and the policy's own blanket `policy_exception_clause`.

### `src/elodie/` (new package)
- **`rules.py`** -- `SleecRule` / `Defeater` dataclasses; `load_rules()` and `get_rule(rule_id)` to read `data/sleec_rules.json`.
- **`prompt_generation.py`** -- Phase 1 (Test Generator) implementation:
  - Calls Gemini via LangChain (`ChatGoogleGenerativeAI`, default model `gemini-2.5-pro`) using `.with_structured_output()` against a `GeneratedPrompts` Pydantic model, so the response is a validated `{"prompts": [...]}` array rather than free text.
  - `generate_prompts(rule, n)` -- one API call generates all `n` prompts together; retries once on a count mismatch.
  - `save_prompts(rule, prompts, output_path)` -- writes one JSON file per rule.
  - Raises if called on a rule flagged `excluded_from_generation`.

### `src/generate_initial_prompts.py` (new CLI)
- Runs with **no arguments**: generates 10 prompts for every rule that isn't excluded (17 of 20), saving to `outputs/prompts/<rule_id>_prompts.json`.
- Flags: `--rule-id`, `--all`, `--num-prompts/-n`, `--model`, `--output-dir`, `--rules-file`.
- Loads `GOOGLE_API_KEY` via `load_dotenv()`; warns if unset.
- Excluded rules print a skip notice (`Skipping <rule_id>: <reason>`) instead of failing.

### `requirements.txt`
New dependencies for the pipeline: `langchain-core`, `langchain-google-genai`, `python-dotenv`.

### `outputs/prompts/` (new, empty)
Destination directory for generated prompt files.

## Not changed
- `src/deberta_example.py` and `README.md` are untouched.

## Open item (paused)
Diversity across separate generation runs (e.g. topping up a rule's prompt count later) isn't handled yet -- each `generate_prompts()` call has no memory of previously saved prompts for that rule, so re-running could duplicate earlier output. Discussed adding an `--append`-style flag that feeds prior prompts back as "don't repeat these" context; not implemented, on pause per request.
