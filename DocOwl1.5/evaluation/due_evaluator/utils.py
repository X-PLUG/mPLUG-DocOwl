from due_evaluator.scorers.fscorer import FScorer
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from due_evaluator.due_evaluator import DueEvaluator


def dataframe_to_print(df: pd.DataFrame, print_format: Optional[str] = 'text') -> str:
    """Export dataframe to json or plain text.

    Args:
        df (pd.DataFrame): data
        print_format (str, optional): Print format. Defaults to 'text'.

    Raises:
        ValueError: unknown print_format

    Returns:
        str: printed version of dataframe

    """
    out: str
    if print_format == 'latex':
        out = df.reset_index().to_latex(index=False)
    elif print_format == 'text':
        out = df.reset_index().to_string(index=False)
    elif print_format == 'json':
        out = df.to_json(orient='index')
    else:
        raise ValueError()
    return out


def property_scores_to_string(
    dues: List[DueEvaluator], print_format: str = 'text', columns: Sequence[str] = ('Precision', 'Recall', 'F-1'),
) -> str:
    """Print out scores per property.

    Args:
        dues: List of DueEvaluators
        print_format: output format: text or latex
        columns: a list of metrics to print

    Returns:
        str: string table with feature scores.

    """
    data = []
    for property_name in sorted(dues[0].property_scorers.keys()) + ['ALL']:
        row_data: Dict[str, Union[str, float]] = {}
        row_data['Label'] = property_name
        for due in dues:
            if len(dues) == 1:
                suffix = ''
            else:
                suffix = f' ({due.path})'
            if property_name == 'ALL':
                scorer = due.general_scorer
            else:
                scorer = due.property_scorers[property_name]

            row_data[scorer.metric_name() + suffix] = scorer.score()
            if isinstance(scorer, FScorer):
                if 'Precision' in columns:
                    row_data['Precision' + suffix] = scorer.precision()
                if 'Recall' in columns:
                    row_data['Recall' + suffix] = scorer.recall()
        data.append(row_data)

    df = pd.DataFrame(data)
    df.set_index('Label', drop=True, inplace=True)

    return dataframe_to_print(df, print_format)
