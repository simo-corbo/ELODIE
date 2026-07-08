"""ELODIE end-to-end CLI, run as decoupled phases with saved artifacts:

    Phase A  SUT interaction     -> outputs/responses.jsonl
    Phase B  oracle scoring      -> outputs/scores.jsonl   (all gates, all branches, no thresholds)
    Phase C  post-hoc thresholds -> outputs/results.csv + summary.json

Two input modes feed Phase A:

  * prompt-centric (default when a prompts corpus is configured): read real
    prompts that already carry per-rule Gate-1 scores and test each against the
    rules it triggers — no generation, Gate 1 is free.
  * rule-centric (--generate, or no corpus): generate adversarial prompts per
    rule via Gemini.

    python run_pipeline.py                          # full run, prompt-centric
    python run_pipeline.py --rule-id SELF_HARM      # only that rule
    python run_pipeline.py --max-prompts 200        # cap SUT calls (strongest first)
    python run_pipeline.py --max-prompts 200 --stratify  # balanced sample across rules
    python run_pipeline.py --generate --num-prompts 5    # rule-centric generation
    python run_pipeline.py --sut deepseek

    # Resume from a saved artifact (thresholds come from --config):
    python run_pipeline.py --from-responses outputs/responses.jsonl  # re-score, skip the SUT
    python run_pipeline.py --from-scores    outputs/scores.jsonl     # only re-resolve thresholds

Because scores.jsonl is threshold-free, --from-scores lets you pick tau_1..tau_4
post-hoc: edit config/elodie.yaml and re-resolve in seconds (no SUT, no oracle).

Reads GOOGLE_API_KEY (generator + Gemini SUT) and DEEPSEEK_API_KEY / a local
endpoint for other SUTs. Wire your own components via elodie.pipeline directly
if you need something the CLI doesn't expose.
"""

import argparse
import re
from pathlib import Path

from dotenv import load_dotenv

from elodie.config import DEFAULT_CONFIG_PATH, load_config
from elodie.evaluation import load_scores, save_scores
from elodie.oracle import DebertaOracle
from elodie.pipeline import (
    load_responses,
    resolve_scores,
    run_pipeline,
    run_prompt_centric,
    score_responses,
    summarise_run,
)
from elodie.prompt_dataset import load_prompt_records, select_prompts
from elodie.rules import get_rule, load_rules
from generator import build_llm


def _sut_slug(name: str) -> str:
    """Filesystem-safe slug for a SUT name, e.g. 'GeminiClient:gemini-2.5-flash'
    -> 'gemini-gemini-2.5-flash'. Drops the redundant 'client' from the client
    class name. Used to namespace each SUT's outputs so running several SUTs
    never clobbers another's responses/scores/results."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower().replace("client", "")).strip("-")
    return slug or "sut"


def _print_overall(result: dict) -> None:
    overall = result["summary"]["overall"]
    print("\n=== Overall ===")
    print(f"  cases: {overall.get('n_cases', 0)}   valid exploits: {overall.get('n_valid_exploits', 0)}")
    print(f"  EVR: {overall.get('exploit_validity_rate')}   "
          f"RVR: {overall.get('rule_violation_rate')}   "
          f"MAS: {overall.get('mean_adherence_score')}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the full ELODIE evaluation pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help=f"Parameters file (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--rule-id", help="Evaluate a single rule (default: all rules).")
    parser.add_argument("--sut", help="SUT type override: gemini | deepseek | local.")
    parser.add_argument("--output-dir", type=Path, help="Where to write results (overrides config).")
    # Prompt-centric (default when a corpus is configured).
    parser.add_argument("--prompts-file", type=Path,
                        help="Prompt corpus (cached scores JSON) to test; overrides config.")
    parser.add_argument("--max-prompts", type=int,
                        help="Cap prompts tested (strongest-triggering first). Default: all.")
    parser.add_argument("--stratify", action="store_true",
                        help="Fill --max-prompts round-robin across rules (balanced per-rule sample) "
                             "instead of taking the globally strongest-triggering prompts.")
    parser.add_argument("--keep-unassigned", action="store_true",
                        help="Keep prompts that trigger no rule under test (default: exclude them; "
                             "they produce no cases either way).")
    parser.add_argument("--complete", action="store_true",
                        help="Prompt-centric: score every prompt against ALL rules under test (not "
                             "just the ones it triggers), so Gate-1 non-triggers become recorded "
                             "vacuous_pass cases. Much slower; off by default.")
    # Rule-centric (generation) fallback.
    parser.add_argument("--generate", action="store_true",
                        help="Ignore the corpus and generate prompts per rule via Gemini.")
    parser.add_argument("--num-prompts", "-n", type=int,
                        help="Prompts to generate per rule in --generate mode (overrides config).")
    # Resume from a saved phase artifact (thresholds still come from --config).
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--from-responses", type=Path,
                        help="Skip the SUT: load a responses.jsonl and run scoring + resolve.")
    resume.add_argument("--from-scores", type=Path,
                        help="Skip SUT + oracle: load a scores.jsonl and only re-resolve verdicts "
                             "at the config thresholds (post-hoc re-thresholding).")
    args = parser.parse_args()

    config = load_config(args.config)

    # -- Post-hoc re-threshold: no SUT, no oracle -- just resolve saved scores. --
    # Results land next to the input scores (its per-SUT dir) unless overridden.
    if args.from_scores:
        out = args.output_dir or Path(args.from_scores).parent
        print(f"Resolving {args.from_scores} at thresholds {vars(config.thresholds)}...")
        scores = load_scores(args.from_scores)
        cases = resolve_scores(scores, config.thresholds)
        result = summarise_run(cases, config.thresholds, "(from scores)", out,
                               source=str(args.from_scores))
        _print_overall(result)
        return

    rules = [get_rule(args.rule_id, config.rules_path)] if args.rule_id else load_rules(config.rules_path)

    # -- Resume from saved SUT responses: skip the SUT, run scoring + resolve. --
    if args.from_responses:
        out = args.output_dir or Path(args.from_responses).parent
        print(f"Loading oracle ({config.oracle_model})...")
        oracle = DebertaOracle(model_name=config.oracle_model)
        responses = load_responses(args.from_responses)
        rules_by_id = {r.rule_id: r for r in rules}
        print(f"Scoring {len(responses)} saved responses (thresholds {vars(config.thresholds)})...")
        scores = score_responses(oracle, responses, rules_by_id,
                                 prompt_scores_path=out.parent / "prompt_scores.jsonl")
        save_scores(scores, out / "scores.jsonl")
        cases = resolve_scores(scores, config.thresholds, rule_ids=list(rules_by_id))
        result = summarise_run(cases, config.thresholds, "(from responses)", out,
                               source=str(args.from_responses))
        _print_overall(result)
        return

    # -- Full run: SUT + oracle. --
    if args.sut:
        config.sut["type"] = args.sut
    prompts_file = args.prompts_file or (None if args.generate else config.prompts_path)

    print(f"Loading oracle ({config.oracle_model})...")
    oracle = DebertaOracle(model_name=config.oracle_model)
    sut = build_llm(config.sut)
    # Namespace this run's artifacts by SUT so multiple SUTs never clobber.
    output_dir = (args.output_dir or config.output_dir) / _sut_slug(sut.name)
    print(f"SUT: {sut.name}   thresholds: {vars(config.thresholds)}")
    print(f"Output dir: {output_dir}\n")

    if prompts_file is not None:
        if not Path(prompts_file).exists():
            parser.error(f"prompts file not found: {prompts_file} (use --generate to skip it)")
        print(f"Prompt-centric: loading corpus {prompts_file}...")
        records = load_prompt_records(prompts_file)
        rule_ids = [r.rule_id for r in rules]
        selected = select_prompts(
            records, config.thresholds.exploit, rule_ids=rule_ids, limit=args.max_prompts,
            exclude_unassigned=not args.keep_unassigned, stratify=args.stratify,
        )
        cap = (f", {'stratified ' if args.stratify else ''}capped to {args.max_prompts}"
               if args.max_prompts else "")
        kept = "kept" if args.keep_unassigned else "excluded"
        print(f"  {len(records)} prompts in corpus -> {len(selected)} selected "
              f"(tau_1={config.thresholds.exploit}, unassigned {kept}{cap}).\n")
        result = run_prompt_centric(
            oracle=oracle,
            rules=rules,
            records=selected,
            sut=sut,
            thresholds=config.thresholds,
            output_dir=output_dir,
            complete=args.complete,
        )
    else:
        from elodie.prompt_generation import generate_prompts

        num_prompts = args.num_prompts or config.generator.get("num_prompts", 10)
        gen_model = config.generator.get("model", "gemini-2.5-pro")
        print(f"Rule-centric: generating {num_prompts} prompts/rule via {gen_model}.\n")

        def generate_fn(rule):
            return generate_prompts(rule, num_prompts, model=gen_model)

        result = run_pipeline(
            oracle=oracle,
            rules=rules,
            generate_fn=generate_fn,
            sut=sut,
            thresholds=config.thresholds,
            output_dir=output_dir,
        )

    _print_overall(result)


if __name__ == "__main__":
    main()
