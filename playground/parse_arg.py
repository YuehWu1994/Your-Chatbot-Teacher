import argparse

def parse_args():
	parser.argparse.Argumentparser()
	parser.add_argument('--emb', action='store', dest='embedding_path',help='path to word embedding file')
	parser.add_argument('--data', action='store', dest='data_path', help='path to data')

	return parser.parse_args()