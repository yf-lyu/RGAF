import argparse, os, logging
import torch, random
import numpy as np
from processor.dataset import MyDataset, PadCollate
from torch.utils.data import DataLoader
from model.build_model import MSD_Knowledge_Model, MSD_Noknowledge_Model
from module.train import MSDTrainer
from transformers import BertTokenizer, CLIPTokenizer, CLIPImageProcessor


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def set_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--bert_name', type=str, default='bert-base-uncased', help='choose bert name')
    parser.add_argument('--vit_name', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--clip_name', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--pretrained_lr', type=float, default=3e-5)
    parser.add_argument('--clip_lr', type=float, default=5e-6)
    parser.add_argument('--other_lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--device', default='cpu', type=str, help="cuda or cpu")
    parser.add_argument('--text_path', type=str, default='twitter/dataset_text')
    parser.add_argument('--img_path', type=str, default='twitter/dataset_image')
    parser.add_argument('--save_model_path', type=str, default='model_ckpt_path')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2023, help="random seed, default is 2023")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--clip_knowledge', action='store_true')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--queue_size', type=int, default=1024)
    parser.add_argument('--momentum', type=float, default=0.995)
    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--ce_loss_weight', type=float, default=1.0)
    parser.add_argument('--global_loss_weight', type=float, default=1.0)
    parser.add_argument('--token_loss_weight', type=float, default=0.1)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--shared_space_dim', type=int, default=400)
    return parser.parse_args()


def set_seed(seed=2023):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = set_parse()
    print(args)

    text_train_path = os.path.join(args.text_path, 'traindep.json')
    text_val_path = os.path.join(args.text_path, 'valdep.json')
    text_test_path = os.path.join(args.text_path, 'testdep.json')

    img_path = args.img_path

    set_seed(args.seed)

    if args.clip_knowledge:
        model = MSD_Knowledge_Model(args=args)
    else:
        model = MSD_Noknowledge_Model(args=args)

    model.to(args.device)

    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_name)
    clip_imgprocess = CLIPImageProcessor.from_pretrained(args.clip_name)

    train_dataset = MyDataset(
        text_path=text_train_path,
        img_path=img_path,
        mode='train',
        args=args
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=PadCollate(
            args=args,
            tokenizer=tokenizer,
            clip_tokenizer=clip_tokenizer,
            clip_imgprocess=clip_imgprocess
        ),
        drop_last=True
    )
    print("train dataset has been loaded successful!")

    val_dataset = MyDataset(
        text_path=text_val_path,
        img_path=img_path,
        mode='val',
        args=args
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=PadCollate(
            args=args,
            tokenizer=tokenizer,
            clip_tokenizer=clip_tokenizer,
            clip_imgprocess=clip_imgprocess
        )
    )
    print("valid dataset has been loaded successful!")

    test_dataset = MyDataset(
        text_path=text_test_path,
        img_path=img_path,
        mode='test',
        args=args
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=PadCollate(
            args=args,
            tokenizer=tokenizer,
            clip_tokenizer=clip_tokenizer,
            clip_imgprocess=clip_imgprocess
        )
    )
    print("test dataset has been loaded successful!")

    trainer = MSDTrainer(train_loader, val_loader, test_loader, model, args, logger)

    # 本地debug专用
    # trainer.train()

    if args.do_train:
        trainer.train()
        args.load_model_path = os.path.join(args.save_model_path, 'best_model.pth')
        trainer.test()
    else:
        trainer.before_train()
        args.load_model_path = os.path.join(args.save_model_path, 'best_model.pth')
        trainer.test()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
