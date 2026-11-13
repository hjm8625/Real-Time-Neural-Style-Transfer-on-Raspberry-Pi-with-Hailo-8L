"""Train the style transfer model with VGG perceptual losses."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import VGG16_Weights, vgg16
from torchvision.utils import save_image

from model import StyleTransferNet


# Training settings
DATA_DIR = Path("data/val2017")
STYLE_IMAGE = Path("images/starry_night.jpg")
OUTPUT_DIR = Path("outputs")

TRAIN_SIZE = 3000
VAL_SIZE = 100

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3

CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 5.0

CONTENT_LAYER = 15  # relu3_3

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_FEATURES
         # using style layers (relu1_2, relu2_2, relu3_3, relu4_3->22)
        self.features = vgg16(weights=weights).features[:23] 
        # VGG is only used to evaluate sytlenet
        self.features.eval()
        self.features.requires_grad_(False)

    def forward(self, x):
        # output of out model is [-1,1]->move to [0,1]
        x = (x + 1.0) / 2.0
        mean = x.new_tensor([0.48235, 0.45882, 0.40784]).view(1, 3, 1, 1)
        std = x.new_tensor([1 / 255, 1 / 255, 1 / 255]).view(1, 3, 1, 1)
        x = (x - mean) / std

        extracted_features = {}

        # extract feature maps from selected layers
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (3, 8, 15, 22):  # relu1_2, relu2_2, relu3_3, relu4_3
                extracted_features[index] = x

        return extracted_features

# gram matrix shows that image’s style, such as colors, textures 
# by calculating correlations between VGG feature channels as a C × C matrix.
def gram_matrix(features: torch.Tensor):
    # [B,C,H,W]
    batch, channels, height, width = features.shape
    # flatten as [B,C, H*W]
    flattened = features.reshape(batch, channels, height * width)
    # bmm([B,C,H*W] x [B,H*W,C]) -> [B,C,C]
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    return gram / (channels * height * width)


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


dataset = datasets.ImageFolder(DATA_DIR.parent, transform=IMAGE_TRANSFORM)

train_dataset, val_dataset, _ = random_split(
    dataset,
    [TRAIN_SIZE, VAL_SIZE, len(dataset) - TRAIN_SIZE - VAL_SIZE], 
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = StyleTransferNet().to(device)
vgg = VGGFeatures().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

with Image.open(STYLE_IMAGE) as image:
    style_image = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device) #[1,3,256,256]

# before training, calculate vgg feature and gram matrix form style_image
with torch.no_grad():
    style_features = vgg(style_image)
    style_gram_targets = {}

    for layer, features in style_features.items():
        style_gram_targets[layer] = gram_matrix(features)

# First sample input to watch results
sample_input = val_dataset[0][0].unsqueeze(0).to(device)

# environmnet print
print(f"Device: {device}")
print(f"Training images: {len(train_dataset)}")

# train
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_index, (content_images, _) in enumerate(train_loader):
        content_images = content_images.to(device)

        generated_images = model(content_images)
        generated_features = vgg(generated_images)

        with torch.no_grad():
            content_features = vgg(content_images)

        content_loss = loss_function(
            generated_features[CONTENT_LAYER],
            content_features[CONTENT_LAYER],
        )

        style_loss = generated_images.new_zeros(())
        for layer in style_gram_targets:
            generated_gram = gram_matrix(generated_features[layer])
            target_gram = style_gram_targets[layer].expand_as(generated_gram)  #[1,C,C] -> [B,C,C]
            style_loss += loss_function(
                generated_gram, 
                target_gram
            )

        loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * content_images.size(0) # batch size

    average_loss = epoch_loss / len(train_dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS}, loss: {average_loss:.6f}")

    model.eval()
    with torch.no_grad():
        sample_output = model(sample_input)

    save_image(
        # output of model is Tanh [-1,1] -> [0,1]
        (sample_output + 1.0) / 2.0,
        OUTPUT_DIR / f"sample_epoch_{epoch + 1:03d}.jpg",
    )
    torch.save(
        model.state_dict(),
        OUTPUT_DIR / f"model_epoch_{epoch + 1:03d}.pt",
    )
