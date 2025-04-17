import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed


class Profile():

    def __init__(self, config):
        # note : if we do not set use_profile, it will be set as None, so that all function will be skip
        self.config = config
        self.skip_prof = False
        self.saved = False
        self.prof = None
        self.rank = torch.distributed.get_rank()
        if config.use_profile and self.rank in self.config.profile_ranks:
            self.prof = torch.profiler.profile(schedule=torch.profiler.schedule(
                wait=max(self.config.step_start - 1, 0),
                warmup=1 if self.config.step_start > 0 else 0,
                active=self.config.step_end - self.config.step_start,
                repeat=1),
                                               record_shapes=True,
                                               with_stack=True)

    def _validate(self):
        pass

    def check(self):
        if self.prof is not None and not self.skip_prof:
            return True
        return False

    def start(self):
        if self.check():
            self.prof.start()

    def step(self):
        if self.check():
            self.prof.step()

    def stop(self):
        if self.check():
            self.stop()

    def save(self):
        if self.check() and not self.save:
            self.saved = True
            save_file_name = f"/prof_{self.config.step_start}_{self.config.step_end}_rank_{self.rank}.json"
            self.prof.export_chrome_trace(self.save_path + save_file_name)
            self.prof.skip_prof()

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()

    def stop_trace(self):
        self.skip_prof = True
