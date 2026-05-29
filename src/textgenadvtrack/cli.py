import argparse
from pathlib import Path

from textgenadvtrack.config import default_model, default_provider, load_project_env
from textgenadvtrack.data.bootstrap_training_data import bootstrap_training_data
from textgenadvtrack.data.append_batches import append_training_batch
from textgenadvtrack.data.build_training_splits import SplitInputs, build_small_validation_splits
from textgenadvtrack.data.collect_human import collect_human_data
from textgenadvtrack.data.augment import build_adversarial_training_rows
from textgenadvtrack.data.complete import CompletedDatasetInputs, build_completed_detection_dataset
from textgenadvtrack.data.external import ExternalDetectionDatasetInputs, ingest_external_detection_dataset
from textgenadvtrack.data.official_splits import OfficialSplitInputs, build_official_detection_splits
from textgenadvtrack.data.public_data import build_merged_training_pool, ingest_public_data
from textgenadvtrack.data.generate_ai import generate_ai_original_data
from textgenadvtrack.data.prepare_official import prepare_official_data
from textgenadvtrack.data.rewrite_ai import generate_ai_rewritten_data
from textgenadvtrack.detection.data_prep import summarize_detection_split
from textgenadvtrack.detection.calibration import tune_scores, tune_submission_scores
from textgenadvtrack.detection.ensemble import blend_prediction_files, search_blend_weights
from textgenadvtrack.detection.validation import (
    build_kfold_detection_splits,
    build_repeated_detection_splits,
    evaluate_prediction_slices,
)
from textgenadvtrack.detection.predict import export_detection_scores, export_detection_submission
from textgenadvtrack.detection.submission_validation import validate_detection_submission
from textgenadvtrack.detection.train import train_detector
from textgenadvtrack.detection.train import supported_backbones, supported_training_backends
from textgenadvtrack.evasion.export import build_evasion_submission
from textgenadvtrack.evasion.generate_candidates import generate_candidates
from textgenadvtrack.evasion.select_candidates import score_and_select_candidates
from textgenadvtrack.evasion.validation import validate_evasion_submission


def build_parser() -> argparse.ArgumentParser:
    load_project_env()
    parser = argparse.ArgumentParser(description="Dual-track baseline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare-official-data")
    prepare_parser.add_argument("--raw-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    bootstrap_parser = subparsers.add_parser("bootstrap-training-data")
    bootstrap_parser.add_argument("--template-dir", type=Path, required=True)
    bootstrap_parser.add_argument("--output-dir", type=Path, required=True)
    append_parser = subparsers.add_parser("append-training-batch")
    append_parser.add_argument("--existing-csv", type=Path, required=True)
    append_parser.add_argument("--batch-csv", type=Path, required=True)
    append_parser.add_argument("--output-csv", type=Path)
    append_parser.add_argument("--dedupe-on", nargs="+")
    collect_human_parser = subparsers.add_parser("collect-human-data")
    collect_human_parser.add_argument("--input-csv", type=Path, required=True)
    collect_human_parser.add_argument("--output-csv", type=Path, required=True)
    collect_human_parser.add_argument("--source-name", required=True)
    collect_human_parser.add_argument("--language", choices=["zh", "en", "ru"], required=True)
    collect_human_parser.add_argument("--domain", required=True)
    collect_human_parser.add_argument("--text-column", default="text")
    external_parser = subparsers.add_parser("ingest-external-detection-dataset")
    external_parser.add_argument("--input-path", type=Path, required=True)
    external_parser.add_argument("--output-csv", type=Path, required=True)
    external_parser.add_argument("--source-name", required=True)
    external_parser.add_argument("--language", choices=["zh", "en", "ru"], required=True)
    external_parser.add_argument("--domain", required=True)
    external_parser.add_argument("--split", choices=["train", "dev", "rewrite_dev"], default="train")
    external_parser.add_argument("--text-column", default="text")
    external_parser.add_argument("--label-column", default="label")
    external_parser.add_argument("--source-model", default="")
    external_parser.add_argument("--prompt-type", default="")
    external_parser.add_argument("--prompt-id-prefix", default="external")
    external_parser.add_argument("--sample-prefix")
    ingest_public_parser = subparsers.add_parser("ingest-public-data")
    ingest_public_parser.add_argument("--input-csv", type=Path, required=True)
    ingest_public_parser.add_argument("--output-csv", type=Path, required=True)
    ingest_public_parser.add_argument("--source-name", required=True)
    ingest_public_parser.add_argument("--language", choices=["zh", "en", "ru"], required=True)
    ingest_public_parser.add_argument("--domain", required=True)
    ingest_public_parser.add_argument("--text-column", default="text")
    ingest_public_parser.add_argument("--text-type", choices=["human", "ai_original", "ai_rewritten"], default="human")
    ingest_public_parser.add_argument("--label", type=int, default=1)
    ingest_public_parser.add_argument("--source-model", default="")
    ingest_public_parser.add_argument("--prompt-type", default="")
    ingest_public_parser.add_argument("--prompt-id-prefix", default="public")
    ingest_public_parser.add_argument("--sample-prefix")
    generate_ai_parser = subparsers.add_parser("generate-ai-original")
    generate_ai_parser.add_argument("--prompt-registry-csv", type=Path, required=True)
    generate_ai_parser.add_argument("--output-csv", type=Path, required=True)
    generate_ai_parser.add_argument("--provider", choices=["mock", "openai"], default=default_provider())
    generate_ai_parser.add_argument("--model-name", default=default_model())
    generate_ai_parser.add_argument("--language", choices=["zh", "en", "ru"])
    generate_ai_parser.add_argument("--max-prompts", type=int)
    rewrite_ai_parser = subparsers.add_parser("generate-ai-rewritten")
    rewrite_ai_parser.add_argument("--ai-original-csv", type=Path, required=True)
    rewrite_ai_parser.add_argument("--rewrite-registry-csv", type=Path, required=True)
    rewrite_ai_parser.add_argument("--output-csv", type=Path, required=True)
    rewrite_ai_parser.add_argument("--provider", choices=["mock", "openai"], default=default_provider())
    rewrite_ai_parser.add_argument("--model-name", default=default_model())
    rewrite_ai_parser.add_argument("--max-rows", type=int)
    build_parser_cmd = subparsers.add_parser("build-training-splits")
    build_parser_cmd.add_argument("--human-csv", type=Path, required=True)
    build_parser_cmd.add_argument("--ai-original-csv", type=Path, required=True)
    build_parser_cmd.add_argument("--ai-rewritten-csv", type=Path, required=True)
    build_parser_cmd.add_argument("--output-dir", type=Path, required=True)
    build_parser_cmd.add_argument("--profile", choices=["small", "seed"], default="small")
    official_split_parser = subparsers.add_parser("build-official-detection-splits")
    official_split_parser.add_argument("--val-with-label-csv", type=Path, required=True)
    official_split_parser.add_argument("--output-dir", type=Path, required=True)
    official_split_parser.add_argument("--dev-fraction", type=float, default=0.2)
    official_split_parser.add_argument("--language", choices=["zh", "en", "ru"], default="zh")
    repeated_split_parser = subparsers.add_parser("build-repeated-detection-splits")
    repeated_split_parser.add_argument("--val-with-label-csv", type=Path, required=True)
    repeated_split_parser.add_argument("--output-dir", type=Path, required=True)
    repeated_split_parser.add_argument("--seeds", type=int, nargs="+", required=True)
    repeated_split_parser.add_argument("--dev-fraction", type=float, default=0.2)
    repeated_split_parser.add_argument("--language", choices=["zh", "en", "ru"], default="zh")
    kfold_split_parser = subparsers.add_parser("build-kfold-detection-splits")
    kfold_split_parser.add_argument("--val-with-label-csv", type=Path, required=True)
    kfold_split_parser.add_argument("--output-dir", type=Path, required=True)
    kfold_split_parser.add_argument("--folds", type=int, default=10)
    kfold_split_parser.add_argument("--seed", type=int, default=42)
    kfold_split_parser.add_argument("--language", choices=["zh", "en", "ru"], default="zh")
    complete_parser = subparsers.add_parser("build-completed-detection-dataset")
    complete_parser.add_argument("--official-val-with-label-csv", type=Path, required=True)
    complete_parser.add_argument("--output-dir", type=Path, required=True)
    complete_parser.add_argument("--dev-fraction", type=float, default=0.2)
    complete_parser.add_argument("--seed", type=int, default=42)
    complete_parser.add_argument("--language", default="unknown")
    augment_parser = subparsers.add_parser("build-adversarial-training-rows")
    augment_parser.add_argument("--detection-csv", type=Path, required=True)
    augment_parser.add_argument("--output-csv", type=Path, required=True)
    augment_parser.add_argument("--max-rows", type=int)
    merge_pool_parser = subparsers.add_parser("build-merged-training-pool")
    merge_pool_parser.add_argument("--human-csv", type=Path, action="append", default=[])
    merge_pool_parser.add_argument("--ai-original-csv", type=Path, action="append", default=[])
    merge_pool_parser.add_argument("--ai-rewritten-csv", type=Path, action="append", default=[])
    merge_pool_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate-detection")
    validate_parser.add_argument("--csv-path", type=Path, required=True)
    train_parser = subparsers.add_parser("train-detector")
    train_parser.add_argument(
        "--model-name",
        required=True,
        help=f"Known backbone name ({', '.join(supported_backbones())}) or a local model directory.",
    )
    train_parser.add_argument("--backend", choices=supported_training_backends(), default="classic")
    train_parser.add_argument("--train-csv", type=Path, required=True)
    train_parser.add_argument("--dev-csv", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=float, default=2.0)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--eval-batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--max-length", type=int, default=512)
    train_parser.add_argument("--max-steps", type=int, default=-1)
    train_parser.add_argument("--no-fp16", action="store_true")
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    predict_parser = subparsers.add_parser("export-detection-submit")
    predict_parser.add_argument("--input-csv", type=Path, required=True)
    predict_parser.add_argument("--model-dir", type=Path, required=True)
    predict_parser.add_argument("--output-xlsx", type=Path, required=True)
    score_parser = subparsers.add_parser("score-detection-csv")
    score_parser.add_argument("--input-csv", type=Path, required=True)
    score_parser.add_argument("--model-dir", type=Path, required=True)
    score_parser.add_argument("--output-csv", type=Path, required=True)
    submit_validate_parser = subparsers.add_parser("validate-detection-submit")
    submit_validate_parser.add_argument("--input-csv", type=Path, required=True)
    submit_validate_parser.add_argument("--submission-xlsx", type=Path, required=True)
    submit_validate_parser.add_argument("--min-unique-scores", type=int, default=10)
    slice_eval_parser = subparsers.add_parser("evaluate-detection-slices")
    slice_eval_parser.add_argument("--labels-csv", type=Path, required=True)
    slice_eval_parser.add_argument("--predictions", type=Path, required=True)
    slice_eval_parser.add_argument("--group-columns", nargs="*", default=[])
    blend_search_parser = subparsers.add_parser("search-detection-blend")
    blend_search_parser.add_argument("--labels-csv", type=Path, required=True)
    blend_search_parser.add_argument("--prediction", type=Path, action="append", required=True)
    blend_search_parser.add_argument("--step", type=float, default=0.1)
    blend_parser = subparsers.add_parser("blend-detection-submit")
    blend_parser.add_argument("--prediction", type=Path, action="append", required=True)
    blend_parser.add_argument("--weight", type=float, action="append", required=True)
    blend_parser.add_argument("--output-xlsx", type=Path, required=True)
    tune_parser = subparsers.add_parser("tune-detection-scores")
    tune_parser.add_argument("--labels-csv", type=Path, required=True)
    tune_parser.add_argument("--predictions", type=Path, required=True)
    tune_parser.add_argument("--scale", type=float, nargs="+", default=[0.7, 0.85, 1.0, 1.15, 1.3])
    tune_parser.add_argument("--bias", type=float, nargs="+", default=[-0.05, -0.025, 0.0, 0.025, 0.05])
    apply_tune_parser = subparsers.add_parser("apply-detection-score-tuning")
    apply_tune_parser.add_argument("--input-xlsx", type=Path, required=True)
    apply_tune_parser.add_argument("--output-xlsx", type=Path, required=True)
    apply_tune_parser.add_argument("--scale", type=float, required=True)
    apply_tune_parser.add_argument("--bias", type=float, required=True)
    generate_parser = subparsers.add_parser("generate-evasion")
    generate_parser.add_argument("--source-csv", type=Path, required=True)
    generate_parser.add_argument("--output-csv", type=Path, required=True)
    generate_parser.add_argument("--rewrite-model", default="rule_rewriter")
    select_parser = subparsers.add_parser("select-evasion")
    select_parser.add_argument("--candidates-csv", type=Path, required=True)
    select_parser.add_argument("--output-csv", type=Path, required=True)
    select_parser.add_argument("--model-dir", type=Path)
    submit_parser = subparsers.add_parser("build-evasion-submit")
    submit_parser.add_argument("--official-input-csv", type=Path, required=True)
    submit_parser.add_argument("--selected-csv", type=Path, required=True)
    submit_parser.add_argument("--output-csv", type=Path, required=True)
    evasion_validate_parser = subparsers.add_parser("validate-evasion-submit")
    evasion_validate_parser.add_argument("--official-input-csv", type=Path, required=True)
    evasion_validate_parser.add_argument("--submission-csv", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare-official-data":
        print(prepare_official_data(args.raw_dir, args.output_root))
    elif args.command == "bootstrap-training-data":
        print(bootstrap_training_data(args.template_dir, args.output_dir))
    elif args.command == "append-training-batch":
        print(
            append_training_batch(
                args.existing_csv,
                args.batch_csv,
                output_csv=args.output_csv,
                dedupe_on=args.dedupe_on,
            )
        )
    elif args.command == "collect-human-data":
        print(
            collect_human_data(
                args.input_csv,
                args.output_csv,
                source_name=args.source_name,
                language=args.language,
                domain=args.domain,
                text_column=args.text_column,
            )
        )
    elif args.command == "ingest-external-detection-dataset":
        print(
            ingest_external_detection_dataset(
                ExternalDetectionDatasetInputs(
                    input_path=args.input_path,
                    output_csv=args.output_csv,
                    source_name=args.source_name,
                    language=args.language,
                    domain=args.domain,
                    split=args.split,
                    text_column=args.text_column,
                    label_column=args.label_column,
                    source_model=args.source_model,
                    prompt_type=args.prompt_type,
                    prompt_id_prefix=args.prompt_id_prefix,
                    sample_prefix=args.sample_prefix,
                )
            )
        )
    elif args.command == "ingest-public-data":
        print(
            ingest_public_data(
                args.input_csv,
                args.output_csv,
                source_name=args.source_name,
                language=args.language,
                domain=args.domain,
                text_column=args.text_column,
                text_type=args.text_type,
                label=args.label,
                source_model=args.source_model,
                prompt_type=args.prompt_type,
                prompt_id_prefix=args.prompt_id_prefix,
                sample_prefix=args.sample_prefix,
            )
        )
    elif args.command == "generate-ai-original":
        print(
            generate_ai_original_data(
                args.prompt_registry_csv,
                args.output_csv,
                model_name=args.model_name,
                provider=args.provider,
                language=args.language,
                max_prompts=args.max_prompts,
            )
        )
    elif args.command == "build-merged-training-pool":
        print(
            build_merged_training_pool(
                human_csvs=args.human_csv,
                ai_original_csvs=args.ai_original_csv,
                ai_rewritten_csvs=args.ai_rewritten_csv,
                output_dir=args.output_dir,
            )
        )
    elif args.command == "generate-ai-rewritten":
        print(
            generate_ai_rewritten_data(
                args.ai_original_csv,
                args.rewrite_registry_csv,
                args.output_csv,
                model_name=args.model_name,
                provider=args.provider,
                max_rows=args.max_rows,
            )
        )
    elif args.command == "build-training-splits":
        print(
            build_small_validation_splits(
                SplitInputs(
                    human_csv=args.human_csv,
                    ai_original_csv=args.ai_original_csv,
                    ai_rewritten_csv=args.ai_rewritten_csv,
                    output_dir=args.output_dir,
                    profile=args.profile,
                )
            )
        )
    elif args.command == "build-official-detection-splits":
        print(
            build_official_detection_splits(
                OfficialSplitInputs(
                    val_with_label_csv=args.val_with_label_csv,
                    output_dir=args.output_dir,
                    dev_fraction=args.dev_fraction,
                    language=args.language,
                )
            )
        )
    elif args.command == "build-repeated-detection-splits":
        print(
            build_repeated_detection_splits(
                args.val_with_label_csv,
                args.output_dir,
                seeds=args.seeds,
                dev_fraction=args.dev_fraction,
                language=args.language,
            )
        )
    elif args.command == "build-kfold-detection-splits":
        print(
            build_kfold_detection_splits(
                args.val_with_label_csv,
                args.output_dir,
                folds=args.folds,
                seed=args.seed,
                language=args.language,
            )
        )
    elif args.command == "build-completed-detection-dataset":
        print(
            build_completed_detection_dataset(
                CompletedDatasetInputs(
                    official_val_with_label_csv=args.official_val_with_label_csv,
                    output_dir=args.output_dir,
                    dev_fraction=args.dev_fraction,
                    seed=args.seed,
                    language=args.language,
                )
            )
        )
    elif args.command == "build-adversarial-training-rows":
        print(build_adversarial_training_rows(args.detection_csv, args.output_csv, max_rows=args.max_rows))
    elif args.command == "validate-detection":
        print(summarize_detection_split(args.csv_path))
    elif args.command == "train-detector":
        print(
            train_detector(
                args.train_csv,
                args.dev_csv,
                args.model_name,
                args.output_dir,
                args.backend,
                epochs=args.epochs,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                max_steps=args.max_steps,
                fp16=not args.no_fp16,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                weight_decay=args.weight_decay,
            )
        )
    elif args.command == "export-detection-submit":
        print(export_detection_submission(args.input_csv, args.model_dir, args.output_xlsx))
    elif args.command == "score-detection-csv":
        print(export_detection_scores(args.input_csv, args.model_dir, args.output_csv))
    elif args.command == "validate-detection-submit":
        print(validate_detection_submission(args.input_csv, args.submission_xlsx, args.min_unique_scores))
    elif args.command == "evaluate-detection-slices":
        print(evaluate_prediction_slices(args.labels_csv, args.predictions, args.group_columns))
    elif args.command == "search-detection-blend":
        print(search_blend_weights(args.labels_csv, args.prediction, step=args.step))
    elif args.command == "blend-detection-submit":
        print(blend_prediction_files(args.prediction, args.weight, args.output_xlsx))
    elif args.command == "tune-detection-scores":
        import pandas as pd

        from textgenadvtrack.detection.ensemble import read_prediction_file

        labels = pd.read_csv(args.labels_csv)["label"].astype(int).tolist()
        predictions = read_prediction_file(args.predictions)["text_prediction"].astype(float).tolist()
        print(tune_scores(labels, predictions, scales=args.scale, biases=args.bias))
    elif args.command == "apply-detection-score-tuning":
        print(tune_submission_scores(args.input_xlsx, args.output_xlsx, scale=args.scale, bias=args.bias))
    elif args.command == "generate-evasion":
        rows = generate_candidates(args.source_csv, args.output_csv, args.rewrite_model)
        print({"generated_candidates": len(rows), "output_csv": str(args.output_csv)})
    elif args.command == "select-evasion":
        rows = score_and_select_candidates(args.candidates_csv, args.output_csv, args.model_dir)
        print({"selected_rows": len(rows), "output_csv": str(args.output_csv)})
    elif args.command == "build-evasion-submit":
        print(build_evasion_submission(args.official_input_csv, args.selected_csv, args.output_csv))
    elif args.command == "validate-evasion-submit":
        print(validate_evasion_submission(args.official_input_csv, args.submission_csv))


if __name__ == "__main__":
    main()
