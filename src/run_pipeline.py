"""ELODIE end-to-end CLI: generate -> run SUT -> score gates -> save results.

    python run_pipeline.py                 # all rules, using config/elodie.yaml
    python run_pipeline.py --rule-id SELF_HARM
    python run_pipeline.py --sut deepseek --num-prompts 5

Reads GOOGLE_API_KEY (generator + Gemini SUT) and DEEPSEEK_API_KEY / a local
endpoint for other SUTs. Wire your own components via elodie.pipeline.run_pipeline
if you need something the CLI doesn't expose.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from elodie.config import DEFAULT_CONFIG_PATH, load_config
from elodie.oracle import DebertaOracle
from elodie.pipeline import run_pipeline
from elodie.prompt_generation import generate_prompts
from elodie.rules import get_rule, load_rules
from generator import build_llm


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the full ELODIE evaluation pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help=f"Parameters file (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--rule-id", help="Evaluate a single rule (default: all rules).")
    parser.add_argument("--num-prompts", "-n", type=int,
                        help="Prompts to generate per rule (overrides config).")
    parser.add_argument("--sut", help="SUT type override: gemini | deepseek | local.")
    parser.add_argument("--output-dir", type=Path, help="Where to write results (overrides config).")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.sut:
        config.sut["type"] = args.sut
    output_dir = args.output_dir or config.output_dir
    num_prompts = args.num_prompts or config.generator.get("num_prompts", 10)
    gen_model = config.generator.get("model", "gemini-2.5-pro")

    rules = [get_rule(args.rule_id, config.rules_path)] if args.rule_id else load_rules(config.rules_path)

    print(f"Loading oracle ({config.oracle_model})...")
    oracle = DebertaOracle(model_name=config.oracle_model)
    sut = build_llm(config.sut)
    print(f"SUT: {sut.name}   thresholds: {vars(config.thresholds)}\n")

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

    overall = result["summary"]["overall"]
    print("\n=== Overall ===")
    print(f"  cases: {overall.get('n_cases', 0)}   valid exploits: {overall.get('n_valid_exploits', 0)}")
    print(f"  EVR: {overall.get('exploit_validity_rate')}   "
          f"RVR: {overall.get('rule_violation_rate')}   "
          f"MAS: {overall.get('mean_adherence_score')}")


if __name__ == "__main__":
    main()
