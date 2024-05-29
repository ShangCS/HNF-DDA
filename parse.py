from hnformer import *

def parse_method(args, c, d, device):
    model = HNFormer(d, args.input_channels, args.hidden_channels, c, num_layers=args.num_layers,
                       dropout=args.dropout,
                       num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                       use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act,
                       use_jk=args.use_jk,
                       nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans).to(device)
    return model


def parser_add_main_args(parser):
    # dataset, protocol
    parser.add_argument('--method', '-m', type=str, default='hnformer')
    parser.add_argument('--dataset', type=str, default='KEGG')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--cpu', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True, help='whether to save model')

    # hyper-parameter for model arch and training
    parser.add_argument('--input_channels', type=int, default=512)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')

    # hyper-parameter for hnformer
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--M', type=int,
                        default=30, help='number of random features')
    parser.add_argument('--use_gumbel', action='store_true', help='use gumbel softmax for message passing')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_act', action='store_true', help='use non-linearity for each layer')
    parser.add_argument('--use_jk', action='store_true', help='concat the layer-wise results in the final layer')
    parser.add_argument('--K', type=int, default=10, help='num of samples for gumbel softmax sampling')
    parser.add_argument('--tau', type=float, default=0.25, help='temperature for gumbel softmax')
    parser.add_argument('--lamda', type=float, default=0.1, help='weight for edge reg loss')
    parser.add_argument('--beta', type=float, default=0.1, help='weight for subgraph contrastive loss')
    parser.add_argument('--rb_order', type=int, default=0, help='order for relational bias, 0 for not use')
    parser.add_argument('--rb_trans', type=str, default='sigmoid', choices=['sigmoid', 'identity'],
                        help='non-linearity for relational bias')
