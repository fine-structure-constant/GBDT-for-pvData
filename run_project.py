"""Run from workspace root: python run_project.py [--prepare-only]."""
import argparse
import json
from pathlib import Path
from pvdelta.prepare import prepare


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='config.json')
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    config=json.loads((root/args.config).read_text(encoding='utf-8'))
    data,audit,issues,out=prepare(config,root)
    if not args.prepare_only:
        from pvdelta.model import train
        from pvdelta.report import create_report
        results,predictions,importance,summary=train(data,config,out)
        create_report(data,audit,issues,results,predictions,importance,summary,config,out)


if __name__=='__main__':
    main()
