import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import json
import clip
from qwen_vl_utils import process_vision_info


# 仅设置一块可见
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- 路径配置 ---
TRAIN_DATA_ROOT = "data/TrainDataset"
TEST_DATA_ROOT = "data/TestDataset"

# 建议换一个新目录，避免覆盖你之前生成的结果
OUTPUT_DIR = "multimodal_clip_tokens"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCAL_MODEL_PATH = "./models/Qwen2.5-VL-7B-Instruct"


# CLIP 文本编码器配置
CLIP_MODEL_NAME = "ViT-B/32"   # 输出文本特征维度为 512


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class CamouflageDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.img_path = os.path.join(data_root, "Imgs")

        self.img_files = sorted([
            f for f in os.listdir(self.img_path)
            if f.endswith('.jpg')
        ])

        print(f"Found {len(self.img_files)} images in {data_root}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_path, img_file)

        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image)

        return img_tensor, img_file, image


def build_clip_text_encoder():
    print(f"Loading CLIP model for text embedding: {CLIP_MODEL_NAME} ...")
    clip_model, _ = clip.load(CLIP_MODEL_NAME, device=device, jit=False)
    clip_model.eval()
    return clip_model


def encode_text_with_clip(clip_model, text):
    """
    使用 CLIP 的 text encoder 将 caption 编码为 512 维文本向量。
    返回 torch.FloatTensor，避免保存 numpy 导致 torch.load 反序列化报错。
    """
    with torch.no_grad():
        text_tokens = clip.tokenize([text], truncate=True).to(device)
        text_embedding = clip_model.encode_text(text_tokens).float()   # [1, 512]
        text_embedding = text_embedding / (
            text_embedding.norm(dim=-1, keepdim=True) + 1e-8
        )

    return text_embedding.squeeze(0).cpu().float()   # torch tensor [512]


def ensure_pil_image(image):
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, np.ndarray):
        return Image.fromarray(image)

    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in [1, 3]:
            arr = np.transpose(arr, (1, 2, 0))
        arr = arr.astype('uint8')
        return Image.fromarray(arr)

    raise TypeError(f"Unsupported image type: {type(image)}")


def postprocess_caption(caption):
    """
    简单清洗 Qwen 输出，尽量保证最终 caption 是：
    The camouflaged target is a {object}, not the {background}.
    """
    caption = caption.strip()
    caption = caption.replace("\n", " ")
    caption = caption.replace("“", "").replace("”", "")
    caption = caption.replace('"', "").replace("'", "")
    caption = " ".join(caption.split())

    # 如果模型输出了额外解释，从目标模板起始处截取
    key = "The camouflaged target is"
    if key in caption:
        caption = caption[caption.find(key):]

    # 如果模型输出多句话，只保留第一句
    if "." in caption:
        caption = caption.split(".")[0].strip() + "."

    # 如果没有句号，补句号
    if not caption.endswith("."):
        caption += "."

    return caption


def generate_caption(model, processor, image, prompt_text):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt_text
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    caption = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    caption = postprocess_caption(caption)

    return caption


def process_split(split_name, data_root, model, processor, clip_model, prompt_text, output_dir):
    dataset = CamouflageDataset(data_root)

    def collate_fn(batch):
        return batch[0]

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    os.makedirs(output_dir, exist_ok=True)
    captions_dict = {}

    print(f"Processing {split_name} ...")

    for img_tensor, img_file, image in tqdm(loader):
        pil_image = ensure_pil_image(image)
        img_path = os.path.join(data_root, "Imgs", img_file)

        caption = generate_caption(model, processor, pil_image, prompt_text)
        text_embedding = encode_text_with_clip(clip_model, caption)

        base_name = os.path.splitext(img_file)[0]
        save_path = os.path.join(output_dir, f"{base_name}.pt")

        print(f"\nImage: {img_file}")
        print(f"Generated Description: {caption}")

        torch.save({
            "caption": caption,
            "text_embedding": text_embedding,
            "image_path": img_path
        }, save_path)

        captions_dict[img_file] = caption

    with open(os.path.join(output_dir, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, indent=4, ensure_ascii=False)


def main():
    print(f"Loading Qwen2.5-VL model from local path: {LOCAL_MODEL_PATH} ...")

    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(
            f"找不到本地模型路径: {LOCAL_MODEL_PATH}，请检查路径是否正确！"
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)
    model.eval()

    clip_model = build_clip_text_encoder()

    # ============================================================
    # 这里是核心修改：
    # 让 Qwen 输出伪装感知对比提示：
    # The camouflaged target is a {object}, not the {background}.
    # ============================================================
    prompt_text = (
        "Identify the camouflaged foreground target and the surrounding background in the image. "
        "Output exactly one short English sentence using this format: "
        "The camouflaged target is a {object}, not the {background}. "
        "Replace {object} with the target object name and replace {background} with the background name. "
        "Do not use braces. "
        "Do not output any explanation. "
        "Do not mention color, texture, shape, contour, or other visual cues."
    )

    train_output_dir = os.path.join(OUTPUT_DIR, "train")

    process_split(
        split_name="training data",
        data_root=TRAIN_DATA_ROOT,
        model=model,
        processor=processor,
        clip_model=clip_model,
        prompt_text=prompt_text,
        output_dir=train_output_dir
    )

    test_datasets = ["CHAMELEON", "CAMO", "COD10K", "NC4K"]

    for dataset_name in test_datasets:
        dataset_path = os.path.join(TEST_DATA_ROOT, dataset_name)

        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist, skipping...")
            continue

        test_output_dir = os.path.join(OUTPUT_DIR, dataset_name)

        process_split(
            split_name=f"{dataset_name} test data",
            data_root=dataset_path,
            model=model,
            processor=processor,
            clip_model=clip_model,
            prompt_text=prompt_text,
            output_dir=test_output_dir
        )

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()