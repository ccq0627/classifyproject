from argparse import ArgumentParser, Namespace
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from PIL import Image

import torch
import matplotlib.pyplot as plt

def make_prediction(args: Namespace) -> None:

    image_path = args.image_path
    device = args.device

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_pil = Image.open(image_path).convert("RGB")

    image = transform(image_pil)  # [3,224,224]
    image = image.unsqueeze(0).to(device)  # [1,3,224,224]

    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    model.heads.head = torch.nn.Linear(in_features=model.heads.head.in_features, out_features=5)
    model.load_state_dict(torch.load("models/best_model.pth"))

    classes = ['ants', 'bees', 'pizza', 'steak', 'sushi']

    model.eval()
    with torch.no_grad():
        out_logits = model(image)  # [1,5]

        pred_probs = torch.softmax(out_logits, dim=1)  # [1,5]
        pred_value, pred_label = torch.max(pred_probs, dim=1)
    pred_probs_list = pred_probs.squeeze_(0).numpy().tolist()
    print(f"Predicted: {classes[pred_label.item()]}, with prob: {pred_value.item()*100:.2f}%")
    print(f"Predicted probs: {pred_probs_list}")

    plt.figure(figsize=(6, 5))
    plt.imshow(image_pil)
    plt.title(f"Predicted: {classes[pred_label.item()]}, with prob: {pred_value.item()*100:.2f}%")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/prediction_result.png")
    plt.show()

if __name__ == "__main__":

    parser = ArgumentParser(description="Make prediction with the best model")
    parser.add_argument(
        "--image_path", 
        type=str, 
        default="make_prediction/OIP-C.jpg", 
        help="Path to the image for prediction"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to use for prediction"
    )
    args = parser.parse_args()

    make_prediction(args)