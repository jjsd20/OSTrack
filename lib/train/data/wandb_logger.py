from collections import OrderedDict

try:
    import wandb
except ImportError:
    raise ImportError(
        'Please run "pip install wandb" to install wandb')


class WandbWriter:
    def __init__(self, exp_name, cfg, output_dir, cur_step=0, step_interval=0, id='47bo4np1', resume_mode="allow"):
        self.wandb = wandb
        self.step = cur_step
        self.interval = step_interval
        
        # Initialize wandb with flexible resume options
        # resume_mode can be "allow", "must", "never", or "auto"
        try:
            wandb.init(project="tracking", name=exp_name, config=cfg, dir=output_dir, id=id, resume=resume_mode)
            # If resuming and we want to continue from a specific step, adjust accordingly
            if wandb.run.step is not None and resume_mode in ["allow", "must"]:
                # If cur_step is provided and is less than wandb.run.step, we're going backwards
                if cur_step > 0 and cur_step < wandb.run.step:
                    print(f"Warning: Starting from step {cur_step} but wandb has step {wandb.run.step}")
                    print("This will create a gap in the wandb logs.")
                    self.step = cur_step
                else:
                    self.step = max(self.step, wandb.run.step)
            else:
                self.step = cur_step
        except Exception as e:
            print(f"Failed to resume wandb run with id {id}: {e}")
            print("Starting a new wandb run...")
            wandb.init(project="tracking", name=exp_name, config=cfg, dir=output_dir, resume="never")
            self.step = cur_step

    def write_log(self, stats: OrderedDict, epoch=-1):
        self.step += 1
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue

            log_dict = {}
            for var_name, val in loader_stats.items():
                if hasattr(val, 'avg'):
                    log_dict.update({loader_name + '/' + var_name: val.avg})
                else:
                    log_dict.update({loader_name + '/' + var_name: val.val})

                if epoch >= 0:
                    log_dict.update({loader_name + '/epoch': epoch})

            # Use self.step directly instead of multiplying by interval to ensure monotonic increase
            # The interval should be handled in the trainer's calling logic, not here
            self.wandb.log(log_dict, step=self.step)
