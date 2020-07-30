import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import *
from rnn import *

class DIN( nn.Module):
    def __init__(self, n_uid, n_mid, n_cid, embedding_dim, ):
        super().__init__()

        self.embedding_layer = InputEmbedding( n_uid, n_mid, n_cid, embedding_dim )
        self.attention_layer = AttentionLayer( embedding_dim, hidden_size = [ 80, 40], activation_layer='sigmoid')
        # self.output_layer = MLP( embedding_dim * 9, [ 200, 80], 1, 'ReLU')
        self.output_layer = MLP( embedding_dim * 7, [ 200, 80], 1, 'ReLU')

    def forward( self, data, neg_sample = False):
                            
        user, material_historical, category_historical, mask, sequential_length , material, category, \
            material_historical_neg, category_historical_neg = data
        
        user_embedding, material_historical_embedding, category_historical_embedding, \
            material_embedding, category_embedding, material_historical_neg_embedding, category_historical_neg_embedding = \
        self.embedding_layer(  user, material, category, material_historical, category_historical, material_historical_neg, category_historical_neg, neg_sample)

        item_embedding = torch.cat( [ material_embedding, category_embedding], dim = 1)
        item_historical_embedding = torch.cat( [ material_historical_embedding, category_historical_embedding], dim = 2 )

        item_historical_embedding_sum = torch.matmul( mask.unsqueeze( dim = 1), item_historical_embedding).squeeze() / sequential_length.type( mask.type() ).unsqueeze( dim = 1)


        attention_feature = self.attention_layer( item_embedding, item_historical_embedding, mask)

        # combination = torch.cat( [ user_embedding, item_embedding, item_historical_embedding_sum, attention_feature ], dim = 1)
        combination = torch.cat( [ user_embedding, item_embedding, item_historical_embedding_sum, 
                                    # item_embedding * item_historical_embedding_sum, 
                                    attention_feature ], dim = 1)

        scores = self.output_layer( combination)

        return scores.squeeze()

class DIEN( nn.Module):
    def __init__(self, n_uid, n_mid, n_cid, embedding_dim):
        super().__init__()

        self.embedding_layer = InputEmbedding( n_uid, n_mid, n_cid, embedding_dim )
        self.gru_based_layer = nn.GRU( embedding_dim * 2 , embedding_dim * 2, batch_first = True)
        self.attention_layer = AttentionLayer( embedding_dim, hidden_size = [ 80, 40], activation_layer='sigmoid')
        self.gru_customized_layer = DynamicGRU( embedding_dim * 2, embedding_dim * 2)

        self.output_layer = MLP( embedding_dim * 9, [ 200, 80], 1, 'ReLU')
        # self.output_layer = MLP( embedding_dim * 9, [ 200, 80], 1, 'ReLU')

    def forward( self, data, neg_sample = False):
                            
        user, material_historical, category_historical, mask, sequential_length , material, category, \
            material_historical_neg, category_historical_neg = data
        
        user_embedding, material_historical_embedding, category_historical_embedding, \
            material_embedding, category_embedding, material_historical_neg_embedding, category_historical_neg_embedding = \
        self.embedding_layer(  user, material, category, material_historical, category_historical, material_historical_neg, category_historical_neg, neg_sample)

        item_embedding = torch.cat( [ material_embedding, category_embedding], dim = 1)
        item_historical_embedding = torch.cat( [ material_historical_embedding, category_historical_embedding], dim = 2 )

        item_historical_embedding_sum = torch.matmul( mask.unsqueeze( dim = 1), item_historical_embedding).squeeze() / sequential_length.unsqueeze( dim = 1)

        output_based_gru, _ = self.gru_based_layer( item_historical_embedding)
        attention_scores = self.attention_layer( item_embedding, output_based_gru, mask, return_scores = True)
        output_customized_gru = self.gru_customized_layer( output_based_gru, attention_scores)

        attention_feature = output_customized_gru[  range( len( sequential_length)), sequential_length - 1]

        combination = torch.cat( [ user_embedding, item_embedding, item_historical_embedding_sum, item_embedding * item_historical_embedding_sum, attention_feature ], dim = 1)

        scores = self.output_layer( combination)

        return scores.squeeze()