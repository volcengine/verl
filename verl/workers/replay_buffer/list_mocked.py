import time

import ray

from verl.workers.replay_buffer.interface import ReplayBufferInterface


class ReplayBuffer(ReplayBufferInterface):
    ray_instance = None

    def __init__(self, remote_deploy=False, max_sample_num=2**15):
        self.store = []
        self.remote_deploy = remote_deploy
        self.max_sample_num = max_sample_num
        self.write_privilege = True

        if remote_deploy:
            self.ray_instance = ray.remote(ReplayBuffer).options(max_concurrency=100).remote(remote_deploy=False, max_sample_num=max_sample_num)

    def put(self, step_sample: tuple, ignore_error=True):
        if self.remote_deploy:
            self.ray_instance.add.remote(step_sample, ignore_error)
        else:
            if len(self.store) < self.max_sample_num:
                while not self.write_privilege:
                    time.sleep(0.1)
                self.store.append(step_sample)
            else:
                raise RuntimeWarning("max capacity reached") if ignore_error else RuntimeError("max capacity reached")

    def get(self, batch_size, short_policy="[]"):
        """
        if len(self.store) < batch_size, short_policy:
        "[]": return [], implemented
        "raise": raise Runtime Error
        "incomplete": return a batch with length less than batch_size
        """
        if self.remote_deploy:
            return ray.get(self.ray_instance.get.remote(batch_size))
        else:
            self.write_privilege = False
            batch = self.store[:batch_size]
            self.store = self.store[batch_size:]
            self.write_privilege = True
            return batch
