import wandb
from utils.metrics import ExplainedClfReport


def create_clf_report_table(
    clf_report: ExplainedClfReport, table_name: str = None
) -> None:
    """
    Reformat an ExplainedClfReport type object into the format required by WANDB.
    """
    my_table = wandb.Table(
        columns=list(clf_report["0"].keys()),
        data=[row.values() for row in clf_report.values()],
    )
    wandb.log({table_name: my_table})
