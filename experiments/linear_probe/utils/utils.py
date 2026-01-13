import accelerate
from torch import nn

def is_main_process() -> bool:
    """
    Check if the current process is the main process.
    """
    return accelerate.PartialState().is_main_process

def print_trainable_parameters(model: nn.Module) -> str:
    '''
    Prints the number of trainable parameters in the model.
    '''
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel # type: ignore
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_params = f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    
    if is_main_process():
        print(trainable_params)
    
    return trainable_params