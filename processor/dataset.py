from torch.utils.data import Dataset
import json, os
from torchvision import transforms
from PIL import Image
import torch
from transformers import CLIPConfig


class MyDataset(Dataset):
    def __init__(self, text_path, img_path, mode, args):
        self.args = args
        self.mode = mode
        self.text_path = text_path
        self.img_path = img_path
        with open(self.text_path) as f:
            self.dataset = json.load(f)
        self.transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                            )
        ])

    def __getitem__(self, item):
        sample = self.dataset[item]
        if self.mode == 'train':
            label = sample[2]
            text = sample[3]

        else:
            label = sample[3]
            text = sample[4]

        token_cap = text['token_cap']

        img = os.path.join(self.img_path, sample[0]+'.jpg')
        img = Image.open(img)
        img = self.transform(img)

        return token_cap, img, label

    def __len__(self):
        return len(self.dataset)


class PadCollate:
    def __init__(self, args, tokenizer, clip_tokenizer, clip_imgprocess):
        self.args = args
        self.tokenizer = tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.clip_imgprocess = clip_imgprocess

    def pad_collate(self, batch):
        batch_imgs = list(map(lambda t: t[1].clone().detach(), batch))
        batch_imgs = torch.stack(batch_imgs)

        batch_labels = torch.tensor(list(map(lambda t: t[2], batch)), dtype=torch.long)

        if self.args.clip_knowledge:
            batch_texts = list(map(lambda t: t[0], batch))
            tokens_len = [len(text) for text in batch_texts]
            max_len = max(tokens_len)
            if max_len > CLIPConfig().text_config.max_position_embeddings:
                max_len = CLIPConfig().text_config.max_position_embeddings
            batch_tokens = self.clip_tokenizer(
                text=batch_texts,
                max_length=max_len,
                truncation=True,
                return_tensors='pt',
                padding=True,
                is_split_into_words=True
            )

            return batch_tokens['input_ids'], batch_tokens['attention_mask'], batch_imgs, batch_labels

        else:
            batch_texts = list(map(lambda t: t[0], batch))
            tokens_len = [len(text) for text in batch_texts]
            batch_tokens = self.tokenizer(
                text=batch_texts,
                is_split_into_words=True,
                truncation=True,
                max_length=max(tokens_len),
                return_tensors='pt',
                padding=True
            )

        return batch_tokens['input_ids'], batch_tokens['token_type_ids'], batch_tokens['attention_mask'], \
            batch_imgs, batch_labels

    def __call__(self, batch):
        return self.pad_collate(batch)

