from configparser import ConfigParser
import sys, os
sys.path.append('..')
#import models

class Configurable(object):
    def __init__(self, config_file):
        config = ConfigParser()
        config.read(config_file)
        self._config = config
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        config.write(open(self.config_file,'w'))
        print('Loaded config file sucessfully.')
        # 打印参数
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    @property
    def pretrained_embeddings_file(self):
        return self._config.get('Data','pretrained_embeddings_file')
    @property
    def bert_dir(self):
        return self._config.get('Data','bert_dir')
    @property
    def data_dir(self):
        return self._config.get('Data','data_dir')
    @property
    def train_file(self):
        return self._config.get('Data','train_file')
    @property
    def dev_file(self):
        return self._config.get('Data','dev_file')
    @property
    def test_file(self):
        return self._config.get('Data','test_file')
    @property
    def max_vocab_size(self):
        return self._config.getint('Data','max_vocab_size')

    @property
    def save_dir(self):
        return self._config.get('Save','save_dir')
    @property
    def config_file(self):
        return self._config.get('Save','config_file')
    @property
    def save_model_path(self):
        return self._config.get('Save','save_model_path')
    @property
    def save_vocab_path(self):
        return self._config.get('Save','save_vocab_path')
    @property
    def load_dir(self):
        return self._config.get('Save','load_dir')
    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')
    @property
    def load_vocab_path(self):
        return self._config.get('Save', 'load_vocab_path')

    @property
    def gru_layers(self):
        return self._config.getint('Network','gru_layers')
    @property
    def word_dims(self):
        return self._config.getint('Network','word_dims')
    @property
    def relation_dims(self):
        return self._config.getint('Network','relation_dims')
    @property
    def dropout_emb(self):
        return self._config.getfloat('Network','dropout_emb')
    @property
    def gru_hiddens(self):
        return self._config.getint('Network','gru_hiddens')
    @property
    def mlp_rel_size(self):
        return self._config.getint('Network', 'mlp_rel_size')
    @property
    def hidden_size(self):
        return self._config.getint('Network', 'hidden_size')
    @property
    def dropout_gru_hidden(self):
        return self._config.getfloat('Network','dropout_gru_hidden')
    @property
    def use_structure(self):
        return self._config.getboolean('Network', 'use_structure')
    @property
    def start_layer(self):
        return self._config.getint('Network', 'start_layer')
    @property
    def end_layer(self):
        return self._config.getint('Network', 'end_layer')
    @property
    def get_label(self):
        return self._config.getint('Network','label_size')

    @property
    def L2_REG(self):
        return self._config.getfloat('Optimizer','L2_REG')
    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer','learning_rate')
    @property
    def decay(self):
        return self._config.getfloat('Optimizer','decay')
    @property
    def decay_steps(self):
        return self._config.getint('Optimizer','decay_steps')
    @property
    def beta_1(self):
        return self._config.getfloat('Optimizer','beta_1')
    @property
    def beta_2(self):
        return self._config.getfloat('Optimizer','beta_2')
    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer','epsilon')
    @property
    def clip(self):
        return self._config.getfloat('Optimizer','clip')

    @property
    def train_iters(self):
        return self._config.getint('Run','train_iters')
    @property
    def train_batch_size(self):
        return self._config.getint('Run','train_batch_size')
    @property
    def test_batch_size(self):
        return self._config.getint('Run','test_batch_size')
    @property
    def validate_every(self):
        return self._config.getint('Run','validate_every')
    @property
    def save_after(self):
        return self._config.getint('Run','save_after')
    @property
    def update_every(self):
        return self._config.getint('Run','update_every')
    @property
    def max_edu_len(self):
        return self._config.getint('Network','max_edu_len')
    @property
    def max_tok_len(self):
        return self._config.getint('Network','max_tok_len')