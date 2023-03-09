
import numpy as np
class Genome():

    def __init__(self,num_of_genes=1,gene_length=10):
        if(num_of_genes <=0):
            raise Exception('Num of Genes must be greature than 0')

        self.genes = [self.__get_random_gene(gene_length) for i in range(num_of_genes)]
        
        # self.genes.append([1,0])
        # self.genes.insert(0,[0,random.randint(3,7)])
    @staticmethod
    def get_gene_spec_entry(dtype,min_val,max_val):
        return {'dtype':dtype,'min':min_val,'max':max_val}
    
    @staticmethod
    def get_gene_spec():
        gene_spec ={
            'initial_learning_rate':Genome.get_gene_spec_entry('FLOAT',0.001,0.1),
            '_batch_size':Genome.get_gene_spec_entry('INT',2,64),
            'hop_size':Genome.get_gene_spec_entry('INT',1,1024),
            'embedding_hidden_size':Genome.get_gene_spec_entry('INT',1,512),
            'initializer_range':Genome.get_gene_spec_entry('FLOAT',0.01,1),
            'embedding_dropout_prob':Genome.get_gene_spec_entry('FLOAT',0.01,1),
            'n_conv_encoder':Genome.get_gene_spec_entry('INT',1,10),
            'encoder_conv_filters':Genome.get_gene_spec_entry('INT',1,512),
            'encoder_conv_kernel_sizes':Genome.get_gene_spec_entry('INT',1,16),
            'encoder_conv_dropout_rate':Genome.get_gene_spec_entry('FLOAT',0.01,1),
            'encoder_lstm_units':Genome.get_gene_spec_entry('INT',0,512),
            'n_prenet_layers':Genome.get_gene_spec_entry('INT',0,10),
            'prenet_units':Genome.get_gene_spec_entry('INT',1,256),
            'prenet_dropout_rate':Genome.get_gene_spec_entry('FLOAT',0.01,1),
            'n_lstm_decoder':Genome.get_gene_spec_entry('INT',1,10),
            '_reduction_factor':Genome.get_gene_spec_entry('INT',1,5),
            'decoder_lstm_units':Genome.get_gene_spec_entry('INT',1,1024),
            'attention_dim':Genome.get_gene_spec_entry('INT',1,256),
            'attention_filters':Genome.get_gene_spec_entry('INT',1,64),
            'attention_kernel':Genome.get_gene_spec_entry('INT',1,64),
            '_n_mels':Genome.get_gene_spec_entry('INT',1,200),
            'n_conv_postnet':Genome.get_gene_spec_entry('INT',1,10),
            'postnet_conv_filters':Genome.get_gene_spec_entry('INT',1,512),
            'postnet_conv_kernel_sizes':Genome.get_gene_spec_entry('INT',1,5),
            'postnet_dropout_rate':Genome.get_gene_spec_entry('FLOAT',0.01,1),
            'prenet_units':Genome.get_gene_spec_entry('INT',1,512),
            }
  

        return gene_spec

    def __get_random_gene(self,length):
        return np.random.random(length)