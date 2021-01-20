import configparser

class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print('loading config: %s failed!' % (config_file))
    
        # model settings
        self.seed = conf.getint('Model_Setup', 'seed')
        self.hidden_dim = conf.getint('Model_Setup', 'hidden_dim')
        self.gcn_layer = conf.getint('Model_Setup', 'gcn_layer')
        self.dropout = conf.getfloat('Model_Setup', 'dropout')
        self.epochs = conf.getint('Model_Setup', 'epochs')
        self.lr = conf.getfloat('Model_Setup', 'lr')
        self.weight_decay = conf.getfloat('Model_Setup', 'weight_decay')
        self.batch_size = conf.getint('Model_Setup', 'batch_size')
        self.readout = conf['Model_Setup']['readout']
        self.fold = conf.getint('Model_Setup', 'fold')  # (1...10) fold cross validation
        self.sort = conf['Model_Setup']['sort'] # sort/sample/random sample
        self.gcn_res = conf.getint('Model_Setup', 'gcn_res') # whether to normalize gcn layers
        self.gcn_norm = conf.getint('Model_Setup', 'gcn_norm') # whether to normalize gcn layers
        self.bn = conf.getint('Model_Setup', 'bn') # whether to normalize gcn layers
        self.relu = conf['Model_Setup']['relu'] # whether to use relu
        self.lastrelu = conf.getint('Model_Setup', 'lastrelu') # whether to use relu
        self.pool = conf['Model_Setup']['pool'] # agcn pool method: global/neighbor
        self.tau = conf.getfloat('Model_Setup', 'tau') # agcn node keep percent(=k/node_num)
        self.dnorm = conf.getint('Model_Setup', 'dnorm') # agcn pool method
        self.single_att = conf.getint('Model_Setup', 'single_att') # agcn pool method: global/neighbor
        self.att_out = conf.getint('Model_Setup', 'att_out') # agcn pool method: global/neighbor
        self.softmax = conf['Model_Setup']['softmax'] # agcn pool method: global/neighbor
        self.khop = conf.getint('Model_Setup', 'khop') # agcn pool method: global/neighbor
        self.adj_norm = conf['Model_Setup']['adj_norm'] # agcn pool method: global/neighbor
        self.percent = conf.getfloat('Model_Setup', 'percent') # agcn node keep percent(=k/node_num)
        self.eps = conf.getfloat('Model_Setup', 'eps')
        self.lamda = conf.getfloat('Model_Setup', 'lamda') # agcn node keep percent(=k/node_num)
        self.att_norm = conf.getint('Model_Setup', 'att_norm') # layer number in each agcn block
        self.dnorm_coe = conf.getfloat('Model_Setup', 'dnorm_coe') # agcn pool method: global/neighbor
        self.diffpool_k = conf.getint('Model_Setup', 'diffpool_k')
        self.pool_layers = conf.getint('Model_Setup', 'pool_layers')
        self.hierarchical_num = conf.getint('Model_Setup', 'hierarchical_num') # pooling layer number
        self.gamma = conf.getfloat('Model_Setup', 'gamma')
        self.local_topology = conf.getboolean('Model_Setup', 'local_topology')
        self.with_feature = conf.getboolean('Model_Setup', 'with_feature')
        self.global_topology = conf.getboolean('Model_Setup', 'global_topology')

        # DiffPool Setting
        self.diffPool_max_num_nodes = conf.getint('DiffPool_Setting', 'diffPool_max_num_nodes')
        self.diffPool_num_gcn_layer = conf.getint('DiffPool_Setting', 'diffPool_num_gcn_layer')
        self.diffPool_assign_ratio = conf.getfloat('DiffPool_Setting', 'diffPool_assign_ratio')
        self.diffPool_num_classes = conf.getint('DiffPool_Setting', 'diffPool_num_classes')
        self.diffPool_num_pool = conf.getint('DiffPool_Setting', 'diffPool_num_pool')
        self.diffPool_bn = conf.getboolean('DiffPool_Setting', 'diffPool_bn')
        self.diffPool_bias = conf.getboolean('DiffPool_Setting', 'diffPool_bias')
        self.diffPool_dropout = conf.getfloat('DiffPool_Setting', 'diffPool_dropout')

        # Data setting
        self.num_class = conf.getint('Data_Setting', 'num_class') # number of graph type
        self.feat_dim = conf.getint('Data_Setting', 'feat_dim') # feature dimension
        self.input_dim = conf.getint('Data_Setting', 'input_dim') # input dimension
        self.attr_dim = conf.getint('Data_Setting', 'attr_dim') # attribute dimension
        self.test_number = conf.getint('Data_Setting', 'test_number') # if specified, will overwrite -fold and use the last -test_number graphs as testing data
        

