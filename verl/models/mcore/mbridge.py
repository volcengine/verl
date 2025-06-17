try:
    from mbridge import AutoBridge
    from mbridge.utils.post_creation_callbacks import freeze_moe_router, make_value_model
except ImportError:
    import subprocess
    import sys

    print("mbridge package not found. This package is required for model bridging functionality.")
    print("Install mbridge with `pip install git+https://github.com/ISEEKYAN/mbridge.git --no-deps`")

    def install_mbridge():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/ISEEKYAN/mbridge.git", "--no-deps"])
        except subprocess.CalledProcessError:
            print("Failed to install mbridge")
            raise

    install_mbridge()
    from mbridge import *

__all__ = ["AutoBridge", "make_value_model", "freeze_moe_router"]
