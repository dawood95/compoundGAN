import torch

from pathlib import Path
from comet_ml import Experiment

class Logger:

    def __init__(self, log_dir, project_name, commit_id, comment=None, disabled=True):

        # COMET ML Setup
        experiment = Experiment(project_name=project_name, disabled=disabled)
        experiment.log_parameter('commit_id', commit_id)
        if comment: experiment.log_other('comment', comment)

        # Setup Model Logging Directory
        experiment_name = '%s-%s'%(project_name, str(experiment.id))
        log_dir = Path(log_dir).expanduser() / experiment_name
        if not log_dir.is_dir() and not disabled:
            log_dir.mkdir(0o755)

        self.log_dir    = log_dir
        self.experiment = experiment
        self.disabled   = disabled

    def save(self, name, data):
        # Save given data in log directory if not disabled
        if self.disabled: return
        torch.save(data, (self.log_dir / name).as_posix())
