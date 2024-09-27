#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from typing import Optional, Set
import json

from due_evaluator.due_evaluator import DueEvaluator
from due_evaluator.utils import property_scores_to_string


def parse_args():
    """Parse CLI arguments.

    Returns:
        namespace: namespace with parsed variables.

    """
    parser = argparse.ArgumentParser('Document Understanding Evaluator')
    parser.add_argument(
        '--out-files',
        '-o',
        type=argparse.FileType('r', encoding='utf-8'),
        required=True,
        nargs='+',
        help='Out file to evaluate',
    )
    parser.add_argument(
        '--reference', '-r', type=argparse.FileType('r', encoding='utf-8'), required=True, help='Reference file',
    )
    parser.add_argument('--metric', '-m', type=str, default='F1', choices=['F1', 'MEAN-F1', 'ANLS', 'WTQ', 'GROUP-ANLS'])
    parser.add_argument(
        '--return-score',
        default='F1',
        choices=['F1', 'mean-F1', 'ANLS', 'mean-Precision', 'mean-Recall', 'WTQ'],
        help='Return WR-like mean-F1 score',
    )
    parser.add_argument('--line-by-line', action='store_true', default=False, help='Return retults example-based')
    parser.add_argument(
        '--columns', type=str, nargs='+', default=['Precision', 'Recall', 'F1'], help='Columns',
    )
    parser.add_argument(
        '--print-format',
        default='text',
        type=str,
        choices=['text', 'latex', 'json'],
        help='Print feature table in the given format',
    )
    parser.add_argument('--properties', nargs='+', type=str, help='Property set to be limitted to')
    parser.add_argument(
        '--ignore-case', '-i', action='store_true', default=False, help='Property set to be limitted to',
    )
    return parser.parse_args()


def cli_main(args: argparse.Namespace):
    """CLI main.

    Args:
        args: cli arguments
    """
    reference = [json.loads(line) for line in args.reference]

    evaluators = []
    for out_file in args.out_files:
        predictions = [json.loads(line) for line in out_file]

        property_set: Optional[Set[str]]
        if args.properties:
            property_set = args.properties
        else:
            property_set = None

        evaluators.append(
            DueEvaluator(reference, predictions, property_set, args.ignore_case, out_file.name, args.metric)
        )

    prop_str = property_scores_to_string(evaluators, args.print_format, args.columns)
    if args.print_format != 'json':
        print(prop_str, file=sys.stderr)

    if args.line_by_line:
        for idx, score in enumerate(evaluators[0].line_by_line()):
            print(f'{idx}: {score}', file=sys.stderr)
    return prop_str


def main() -> None:
    """Main."""
    args = parse_args()
    cli_main(args)


if __name__ == '__main__':
    main()
