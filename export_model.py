
from pathlib import Path
from urllib.request import urlretrieve

MODELS = Path(__file__).resolve().parent / "models"
MODELS.mkdir(parents=True, exist_ok=True)

models = {
    "schp-lip-20-int8-static.onnx": (
        "https://huggingface.co/pirocheto/schp-lip-20/"
        "resolve/main/onnx/schp-lip-20-int8-static.onnx"
    )
}

def download(name, url):
    destination = MODELS / name

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {name}")
        return

    print(f"Downloading {name}...")

    temp = destination.with_suffix(".download")

    try:
        urlretrieve(url, temp)
        temp.replace(destination)
        print(f"Saved: {destination}")
    finally:
        if temp.exists():
            temp.unlink()

for name, url in models.items():
    download(name, url)

detr = MODELS / "resnet50_uint8.onnx"

if detr.exists():
    print("detr exists")
else:
    print("model missing")

print("donezo")
