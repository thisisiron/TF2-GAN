from argparse import ArgumentParser, Namespace
from train import train


def main():
    parser = ArgumentParser(description='train model from data')

    parser.add_argument('--batch-size', help='batch size <default: 32>', metavar='INT',
                        type=int, default=32)

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
