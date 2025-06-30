
class Tpu:
    def __init__(self, device_name):
        import torch_xla
        import torch_xla.core.xla_model as xm
        self.torch_xla = torch_xla
        self.xm = xm
        self.device_name = device_name
    
    def is_available(self):
        return True

    def current_device(self):
        return self.xm.xla_device()

    def set_device(self, local_rank):
        return
    
    def get_device_name(self):
        return ""
    
    def empty_cache(self):
        return
    
    def memory_allocated(self):
        return self.xm.get_memory_info()["peak_bytes_used"]

    def memory_reserved(self):
        return self.xm.get_memory_info()["bytes_used"]
    
    def mem_get_info(self):
        mem_info = self.xm.get_memory_info()
        return mem_info["bytes_limit"] - mem_info["bytes_used"], mem_info["bytes_limit"]

    def max_memory_allocated(self):
        return self.xm.get_memory_info()["peak_bytes_used"]

    def max_memory_reserved(self):
        return self.xm.get_memory_info()["bytes_used"]
    
    def get_rng_state(self):
        return self.xm.get_rng_state()
    
    def set_rng_state(self, seed):
        self.xm.set_rng_state(seed)

    def manual_seed(self, seed):
        self.torch_xla.manual_seed(seed)


