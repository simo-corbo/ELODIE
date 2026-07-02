# Changelog

Changes made while building the ELODIE Phase 1 pipeline (initial prompt generation). All items below are new, untracked files -- nothing pre-existing was modified.

## Added

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
