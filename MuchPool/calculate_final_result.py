import argparse
from utils import Result_generator
from config import Config
import os.path as osp

def generate_result_file_name(dataset, config):
    if config.local_topology and config.with_feature and config.global_topology:
        result_file_name = 'training_log_of_%d_epochs_%d_layer.json' % (config.epochs, config.hierarchical_num)
    elif not config.local_topology:
        result_file_name = 'training_log_of_%d_epochs_%d_layer_without_local_topology.json' % (config.epochs, config.hierarchical_num)
    elif not config.with_feature:
        result_file_name = 'training_log_of_%d_epochs_%d_layer_without_feature.json' % (config.epochs, config.hierarchical_num)
    elif not config.global_topology:
        result_file_name = 'training_log_of_%d_epochs_%d_layer_without_global_topology.json' % (config.epochs, config.hierarchical_num)
    full_name = osp.join(osp.dirname(osp.abspath(__file__)), 'result', dataset, result_file_name)
    return full_name

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', default='COLLAB', type=str, help='dataset')
    args = parse.parse_args()
    config_file = osp.join(osp.dirname(osp.abspath(__file__)), 'config', '%s.ini' % args.dataset)
    config = Config(config_file)
    training_process_data_file = generate_result_file_name(args.dataset, config)
    rg = Result_generator(training_process_data_file, args.dataset, config.hierarchical_num, config.local_topology, config.with_feature, config.global_topology)
    rg.generate_acc_std()
    rg.generate_train_loss_curve()
    rg.generate_test_loss_curve()