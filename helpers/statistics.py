import torch
from evidential.models import *
from evidential.save import *
from models import *
from torchviz import make_dot


def std_prob(probabilities):

    # Berechnen Sie die Standardabweichung entlang der Dimension 1
    std_dev = torch.std(probabilities, dim=1)

    return std_dev


def divide_by_total(evidential_outputs, method='der'):
    """
    Compute normalized uncertainty ratios (aleatoric/total, epistemic/total).
    
    Args:
        evidential_outputs: Dict with uncertainty outputs
        method: 'der' or 'sder' - determines which uncertainties to use
    
    Returns:
        aleatoric_by_total, epistemic_by_total: Normalized uncertainty ratios
    """
    if method == 'sder':
        total = evidential_outputs["aleatoric_sder"] + evidential_outputs["epistemic_sder"]
        aleatoric_by_total = evidential_outputs["aleatoric_sder"] / total
        epistemic_by_total = evidential_outputs["epistemic_sder"] / total
    elif method == 'der':
        total = evidential_outputs["aleatoric_der"] + evidential_outputs["epistemic_der"]
        aleatoric_by_total = evidential_outputs["aleatoric_der"] / total
        epistemic_by_total = evidential_outputs["epistemic_der"] / total
    else:
        raise ValueError(f"Unknown method: '{method}'. Supported methods are 'der' and 'sder'.")
    
    return aleatoric_by_total, epistemic_by_total


def visualize_torchviz():
    dummy_evidential_model = EvidentialWrapper().cuda()
    dummy_aarmvsnet_model = AARMVSNetWrapper().cuda()

    dummy_input = torch.randn(1, 32, 128, 160).cuda()
    output = dummy_evidential_model(dummy_input)
    dot = make_dot(output, params=dict(dummy_evidential_model.named_parameters()))
    dot.render("/home/grannemann/Downloads/evidential_model", format="png")

    dummy_input = torch.randn(1, 5, 3, 128, 160).cuda()
    output = dummy_aarmvsnet_model(dummy_input)
    dot = make_dot(output, params=dict(dummy_evidential_model.named_parameters()))
    dot.render("/home/grannemann/Downloads/aarmvsnet_model", format="png")


#TODO AARMVSNet broken
def export_onnx():
    dummy_evidential_model = EvidentialWrapper().cuda()
    dummy_aarmvsnet_model = AARMVSNetWrapper().cuda()

    dummy_input = torch.randn(1, 32, 128, 160).cuda()
    torch.onnx.export(dummy_evidential_model, dummy_input, '/home/grannemann/Downloads/evidential_model.onnx', input_names=["Features"], output_names=["Evidential Parameters"], opset_version=11)
    print("Exported Evidential onnx model.")

    dummy_input = torch.randn(1, 5, 3, 128, 160).cuda()
    torch.onnx.export(dummy_aarmvsnet_model, dummy_input, '/home/grannemann/Downloads/aarmvsnet_model.onnx', input_names=["image"], output_names=["logits"], opset_version=14, verbose=True)
    print("Exported AARMVSNet onnx model.")


if __name__ == "__main__":
    export_onnx()