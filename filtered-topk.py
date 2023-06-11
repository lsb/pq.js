import torch

class FilteredTopKModel(nn.Module):
    def __init__(self):
        super(FilteredTopKModel, self).__init__()
    
    def forward(self, x, filterColumn, filterValue, z, shim, k):
        (topk_values, topk_indices) = torch.topk(
            x + torch.where(filterColumn == filterValue, z, shim) ,
            k.to(torch.int64),
            largest=False)
        return topk_indices

model = FilteredTopKModel()

torch.onnx.export(
    model,
    (
        torch.arange(1e6, dtype=torch.float32),
        torch.arange(1e6, dtype=torch.float32) % 9,
        torch.tensor([0.]),
        torch.tensor([0.]),
        torch.tensor([12345.]),
        torch.tensor([20], dtype=torch.uint8)
    ),
    "filtered-topk.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names = ['input', 'filterColumn', 'filterValue', 'filterZero', 'filterShim','k',],
    output_names = ['output'],
    dynamic_axes={
        "input": {0: "arraylen"},
        "filterColumn": {0: "arraylen"},
    }
)
