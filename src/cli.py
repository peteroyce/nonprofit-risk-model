"""
Command-line interface for the nonprofit risk model pipeline.

Usage
-----
  python -m src.cli download          # Download IRS data
  python -m src.cli preprocess        # Build features and labels
  python -m src.cli train             # Train the XGBoost model
  python -m src.cli train --sample 0.1   # Quick dev run on 10% of data
  python -m src.cli serve             # Start the FastAPI server
  python -m src.cli predict <ein> <name>  # Score a single nonprofit
  python -m src.cli evaluate          # Generate evaluation report
"""

import argparse
import sys


def cmd_download(args: argparse.Namespace) -> None:
    from src.data.download import download_all
    download_all(force=args.force)


def cmd_preprocess(args: argparse.Namespace) -> None:  # noqa: ARG001
    from src.data.preprocess import run_pipeline
    run_pipeline()


def cmd_train(args: argparse.Namespace) -> None:
    from src.models.train import train
    train(sample_frac=args.sample)


def cmd_evaluate(args: argparse.Namespace) -> None:  # noqa: ARG001
    from src.models.evaluate import generate_report
    generate_report()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    import json
    from src.models.predict import predict_risk

    result = predict_risk(
        ein=args.ein,
        name=args.name,
        state=args.state or "UNK",
        explain=args.explain,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nonprofit-risk-model",
        description="End-to-end pipeline for nonprofit risk prediction",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download IRS datasets")
    dl.add_argument("--force", action="store_true", help="Re-download even if cached")
    dl.set_defaults(func=cmd_download)

    # preprocess
    pp = sub.add_parser("preprocess", help="Build features and labels from raw data")
    pp.set_defaults(func=cmd_preprocess)

    # train
    tr = sub.add_parser("train", help="Train the XGBoost model")
    tr.add_argument("--sample", type=float, default=1.0, help="Fraction of data to use (default: 1.0)")
    tr.set_defaults(func=cmd_train)

    # evaluate
    ev = sub.add_parser("evaluate", help="Generate model evaluation report")
    ev.set_defaults(func=cmd_evaluate)

    # serve
    sv = sub.add_parser("serve", help="Start the FastAPI server")
    sv.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    sv.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    sv.add_argument("--reload", action="store_true", help="Enable auto-reload")
    sv.set_defaults(func=cmd_serve)

    # predict
    pr = sub.add_parser("predict", help="Score a single nonprofit from the CLI")
    pr.add_argument("ein", help="EIN (e.g. 53-0196605)")
    pr.add_argument("name", help="Organisation name")
    pr.add_argument("--state", help="2-letter state code")
    pr.add_argument("--explain", action="store_true", help="Include SHAP explanation")
    pr.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


def validate_0(data):
    """Validate: add data validation"""
    return data is not None
