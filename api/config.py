import os

DATA_DIR = os.getenv("DATA_DIR", "/")

USER_SETTING_DIR = os.path.join(DATA_DIR, "users")
VOICE_DIR =  os.path.join(DATA_DIR, "samples")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USER"] = "Docker"


def user_setting_path(user):
    return os.path.join(USER_SETTING_DIR, f"{user}.json")
