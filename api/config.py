import os

USER_SETTING_DIR = "./users"
VOICE_DIR = "./samples"
CHECKPOINT_DIR = "./checkpoints"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USER"] = "Docker"


def user_setting_path(user):
    return os.path.join(USER_SETTING_DIR, f"{user}.json")
