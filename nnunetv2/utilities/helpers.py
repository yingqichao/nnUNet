import torch


def check_if_proceed(info_to_confirm: str):
    """Prompt user for confirmation before proceeding."""
    print(f"Please confirm the following information: \n{info_to_confirm}\n")
    while True:
        answer = input("Are you sure you want to proceed? (y/n): ").strip().lower()
        if answer == "y":
            print("Proceeding...")
            break
        elif answer == "n":
            print("Cancelled.")
            raise ValueError("User Cancelled.")
        else:
            print("Please enter y or n.")

def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
