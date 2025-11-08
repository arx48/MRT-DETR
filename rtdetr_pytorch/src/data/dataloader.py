import torch 
import torch.utils.data as data

from src.core import register


__all__ = ['DataLoader']


@register
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string



@register
def default_collate_fn(items):
    '''default collate_fn
    '''    
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]

@register
def fusion_collate_fn(items):
    samples = torch.stack([x[0] for x in items], dim=0)  # 把图像拼接为tensor
    samples_ir = torch.stack([x[1] for x in items], dim=0)
    targets = [x[2] for x in items]
    # print(f"[DEBUG] samples type: {type(samples)}, sample_ir type: {type(samples_ir)}, targets type: {type(targets)}")
    return samples, samples_ir, targets
