import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from elodie.prompt_generation import DEFAULT_MODEL, generate_prompts, save_prompts
from elodie.rules import DEFAULT_RULES_PATH, get_rule, load_rules

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "prompts"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "ELODIE Phase 1: generate initial prompts that trigger a SLEEC rule's "
            "antecedent, via Gemini, and save them to a file."
        )
    )
    parser.add_argument(
        "--rule-id",
        help="Rule to generate prompts for (see data/sleec_rules.json). If omitted, every rule is run.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate prompts for every rule in the rules file. This is also the default when --rule-id is omitted.",
    )
    parser.add_argument(
        "--num-prompts",
        "-n",
        type=int,
        default=10,
        help="Number of prompts to generate per rule (default: 10).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model id to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated prompt files (default: outputs/prompts).",
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        default=DEFAULT_RULES_PATH,
        help="Path to the SLEEC rules JSON file.",
    )
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: no GOOGLE_API_KEY set, the Gemini API call will likely fail.")

    if args.rule_id:
        rules = [get_rule(args.rule_id, args.rules_file)]
    else:
        rules = load_rules(args.rules_file)

    for rule in rules:
        print(f"Generating {args.num_prompts} prompts for {rule.rule_id}...")
        prompts = generate_prompts(rule, args.num_prompts, model=args.model)
        output_path = args.output_dir / f"{rule.rule_id.lower()}_prompts.json"
        save_prompts(rule, prompts, output_path)
        print(f"  saved {len(prompts)} prompts -> {output_path}")


if __name__ == "__main__":
    main()
