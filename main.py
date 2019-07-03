import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torchvision.transforms import ToTensor


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--cuda', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to train using CUDA")
    parser.add_argument('--model', type=str, help='Number of iterations between each checkpoint')
    parser.add_argument('--upscale-factor', type=int, help='super resolution upscale factor')
    parser.add_argument('--in-image', type=str, help='Input file path')
    parser.add_argument('--out-image', type=str, help='Output file path')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        raise Exception('CUDA Device not found')

    # The image to resize
    input_image = Image.open(args.in_image).convert('RGB')
    input_image = input_image.resize(
        (
            input_image.size[0] * args.upscale_factor,
            input_image.size[1] * args.upscale_factor
        ),
        Image.BICUBIC
    )
    input_tensor = Variable(ToTensor()(input_image))
    input_tensor = input_tensor.view(1, -1, input_image.size[1], input_image.size[0])

    device = torch.device('cuda' if args.cuda else 'cpu')

    model = torch.load(args.model)
    model.eval()
    if args.cuda:
        input_tensor = input_tensor.to(device)
        model = model.to(device)

    output = model(input_tensor)
    if args.cuda:
        # Move back to CPU before doing the image processing
        output = output.cpu()

    output = transforms.ToPILImage()(output.data[0])
    output = output.convert('RGB')
    output.save(args.out_image)


if __name__ == '__main__':
    main()
