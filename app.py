import torch
import gradio as gr
from src.model import DRModel
from torchvision import transforms as T

CHECKPOINT_PATH = "artifacts/checkpoints/run-2025-04-16-23-41-59/epoch=19-step=40-val_loss=0.79-val_acc=0.80-val_kappa=nan.ckpt"
model = DRModel.load_from_checkpoint(CHECKPOINT_PATH, map_location="cpu")

model.eval()

labels = {
    0: "No DR",
    1: "Moderate",
    2: "Severe",
}

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Define the prediction function
def predict(input_img):
    input_img = transform(input_img).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(input_img)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in labels}
    return confidences


# Set up the Gradio app interface
dr_app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Diabetic Retinopathy Detection App",
    description="Welcome to our Diabetic Retinopathy Detection App! \
        This app utilizes deep learning models to detect diabetic retinopathy in retinal images.\
        Diabetic retinopathy is a common complication of diabetes and early detection is crucial for effective treatment.",
    examples=[
        "data/diabetic-retinopathy-dataset/sample/10_left.jpeg",
        "data/diabetic-retinopathy-dataset/sample/10_right.jpeg",
        "data/diabetic-retinopathy-dataset/sample/15_left.jpeg",
        "data/diabetic-retinopathy-dataset/sample/16_right.jpeg",
    ],
)

# Run the Gradio app
if __name__ == "__main__":
    dr_app.launch()
