import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

parser.add_argument('-v', '--verbose', help='Bobeszedu outputokat adok', action='store_true')

bert_parser = subparsers.add_parser("A")
bert_parser.add_argument('-x', '--xerbose', action='store_true',
                         help="Szia Gergo, miben segitsek?")

nn_parser = subparsers.add_parser("B")
nn_parser.add_argument('-y', '--yerbose', action='store_true')

c_parser = subparsers.add_parser("C")

args = parser.parse_args()
print(vars(args))
