import os


def check_os_environ(key, use):
    if key not in os.environ:
        print(f"{key} is not defined in the os variables, it is required for {use}.")
        print(f"Use home directory by default.")
        return os.path.expanduser("~")
    return os.environ[key]


def log_dir():
    checkpoint = check_os_environ("CHECKPOINT", "model checkpointing")
    return checkpoint
