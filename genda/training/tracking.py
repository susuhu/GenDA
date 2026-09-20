"""W&B is imported only when explicitly enabled by the training CLI."""

class Tracking:
    def __init__(self):
        self.backend = None

    @property
    def run(self):
        return None if self.backend is None else self.backend.run

    def enable(self, **kwargs):
        import wandb
        self.backend = wandb
        wandb.init(**kwargs)

    def __getattr__(self, name):
        if self.backend is None:
            raise RuntimeError("Tracking is not enabled")
        return getattr(self.backend, name)

wandb = Tracking()
