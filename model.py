"""@misc{TEDD1104,
  author = "Garc{\'\i}a-Ferrero, Ike},
  title = TEDD1104: Self Driving Car in Video Games,
  year = 2021,
  publisher = GitHub,
  journal = GitHub repository,
  howpublished = https://github.com/ikergarcia1996/Self-Driving-Car-in-Video-Games,
"""

import torch,torchvision
import pytorch_lightning as pl
from typing import List
import torchmetrics
from metrics import *
from general import Keyboard2Controller
def get_cnn():
    model = torchvision.models.efficientnet_b4(pretrained = True)
    #remove classification
    _ = model._modules.popitem()
    model = torch.nn.Sequential(*list(model.children()))

    # test output size 
    features = model(torch.zeros((1,3,270,480),dtype = torch.float32))
    output_size: int = features.reshape(features.size(0),-1).size(1)
    return model,output_size
class EncoderCNN(torch.nn.Module):
    """
    from data [batch*sequence_size,3,270, 480] to [batch,sequence_size,embedded_size]
    """

    def __init__(
        self,
        embedded_size: int,
        dropout_cnn_out: float,
        sequence_size: int = 5,
    ):
        
        super(EncoderCNN, self).__init__()

        self.embedded_size = embedded_size
        self.dropout_cnn_out = dropout_cnn_out

        self.cnn, self.cnn_output_size = get_cnn()

        self.dp = torch.nn.Dropout(p=dropout_cnn_out)
        self.dense = torch.nn.Linear(self.cnn_output_size, self.cnn_output_size)
        self.layer_norm = torch.nn.LayerNorm(self.cnn_output_size, eps=1e-05)

        self.decoder = torch.nn.Linear(self.cnn_output_size, self.embedded_size)
        self.bias = torch.nn.Parameter(torch.zeros(self.embedded_size))
        self.decoder.bias = self.bias
        self.gelu = torch.nn.GELU()
        self.sequence_size = sequence_size

    def forward(self, images: torch.tensor) -> torch.tensor:
        """
        Forward pass
        :param torch.tensor images: Input images [batch_size * sequence_size, 3, 270, 480]
        :return: Output embedding [batch_size, sequence_size, embedded_size]
        """
        features = self.cnn(images)
        features = features.reshape(features.size(0), -1) # batch_size,-1

        
        #Reshapes the features from the CNN into a time distributed format
       

        features = features.view(
            int(features.size(0) / self.sequence_size),
            self.sequence_size,
            features.size(1),
        )

        features = self.dp(features)
        features = self.dense(features)
        features = self.gelu(features)
        features = self.layer_norm(features)
        features = self.decoder(features)
        return features

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias

class PositionalEmbedding(torch.nn.Module):
    """
    from Input features [batch_size, sequence_size, embedded_size]
        to  Output features [batch_size, sequence_size, embedded_size]
    """

    def __init__(
        self, sequence_length: int, d_model: int, dropout: float = 0.1,
    ):
        
        super(PositionalEmbedding, self).__init__()

        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = dropout

        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(self.sequence_length, d_model).float()
        pe.requires_grad = True
        pe = pe.unsqueeze(0)
        self.pe = torch.nn.Parameter(pe)
        torch.nn.init.normal_(self.pe, std=0.02)
        self.LayerNorm = torch.nn.LayerNorm(self.d_model, eps=1e-05)
        self.dp = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        
        pe = self.pe[:, : x.size(1)]
        x = pe + x
        x = self.LayerNorm(x)
        x = self.dp(x)
        return x

class EncoderTransformer(torch.nn.Module):
    """
        

        from features: Input features [batch_size, sequence_length, embedded_size]
        to: Output features [batch_size, d_model]
        """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 1,
        mask_prob: float = 0.2,
        dropout: float = 0.1,
        sequence_length: int = 5,
    ):
        
        super(EncoderTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.mask_prob = mask_prob
        self.dropout = dropout
        self.sequence_length = sequence_length

        cls_token = torch.zeros(1, 1, self.d_model).float()
        cls_token.require_grad = True
        self.clsToken = torch.nn.Parameter(cls_token)
        torch.nn.init.normal_(cls_token, std=0.02)

        self.pe = PositionalEmbedding(
            sequence_length=self.sequence_length + 1, d_model=self.d_model
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        for parameter in self.transformer_encoder.parameters():
            parameter.requires_grad = True

    def forward(self, features: torch.tensor):
        
        if self.training:
            bernolli_matrix = (
                torch.cat(
                    (
                        torch.tensor([1]).float(),
                        (torch.tensor([self.mask_prob]).float()).repeat(
                            self.sequence_length
                        ),
                    ),
                    0,
                )
                .unsqueeze(0)
                .repeat([features.size(0) * self.nhead, 1])
            )
            bernolli_distributor = torch.distributions.Bernoulli(bernolli_matrix)
            sample = bernolli_distributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1)
        else:

            mask = torch.ones(
                features.size(0) * self.nhead,
                self.sequence_length + 1,
                self.sequence_length + 1,
            )

        mask = mask.type_as(features)
        features = torch.cat(
            (self.clsToken.repeat(features.size(0), 1, 1), features), dim=1
        )
        features = self.pe(features)
        # print(f"x.size(): {x.size()}. mask.size(): {mask.size()}")
        features = self.transformer_encoder(
            features.transpose(0, 1), mask=mask
        ).transpose(0, 1)
        return features
class OutputLayer(torch.nn.Module):
    """
        Forward pass

        from x: Input features [batch_size, d_model] if RNN else [batch_size, sequence_length+1, d_model]
        to: Output features [num_classes]
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout_encoder_features: float = 0.2,
        from_transformer: bool = True,
    ):
        
        super(OutputLayer, self).__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.dropout_encoder_features = dropout_encoder_features
        self.dense = torch.nn.Linear(self.d_model, self.d_model)
        self.dp = torch.nn.Dropout(p=dropout_encoder_features)
        self.out_proj = torch.nn.Linear(self.d_model, self.num_classes)
        self.tanh = torch.nn.Tanh()
        self.from_transformer = from_transformer

    def forward(self, x):
        
        if self.from_transformer:
            x = x[:, 0, :]  # Get [CLS] token
        x = self.dp(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dp(x)
        x = self.out_proj(x)
        return x
class Transformer(torch.nn.Module):
    """
    - cnn to extract features from the images
    - a transformer that extracts a representation of the image sequence
    """
    def __init__(
        self,
        embedded_size:int = 512,
        nhead:int = 8,
        num_layers_transformer:int = 4,
        dropout_cnn_out:str =0.3,
        positional_embeddings_dropout:str = 0.1,
        dropout_transformer:str =0.1,
        dropout_encoder_features:str = 0.3,
        mask_prob:str = 0.2,
        control_mode:str = "keyboard",
        sequence_size: int = 5
    ):
        super(Transformer, self).__init__()
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.nhead: int = nhead
        self.num_layers_transformer: int = num_layers_transformer
        self.dropout_cnn_out: float = dropout_cnn_out
        self.positional_embeddings_dropout: float = positional_embeddings_dropout
        self.dropout_transformer: float = dropout_transformer
        self.mask_prob = mask_prob
        self.control_mode = control_mode
        self.dropout_encoder_features = dropout_encoder_features

        self.EncoderCNN: EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,
            dropout_cnn_out=dropout_cnn_out,
            sequence_size=self.sequence_size,
        )
        self.PositionalEncoding = PositionalEmbedding(
            d_model=embedded_size,
            dropout=self.positional_embeddings_dropout,
            sequence_length=self.sequence_size,
        )

        self.EncoderTransformer: EncoderTransformer = EncoderTransformer(
            d_model=embedded_size,
            nhead=nhead,
            num_layers=num_layers_transformer,
            mask_prob=self.mask_prob,
            dropout=self.dropout_transformer,
        )

        self.OutputLayer: OutputLayer = OutputLayer(
            d_model=embedded_size,
            num_classes=9 if self.control_mode == "keyboard" else 2,
            dropout_encoder_features=dropout_encoder_features,
            from_transformer=True,
        )
    def forward(self, x: torch.tensor) -> torch.tensor:
            """
        Forward pass

        :param torch.tensor x: Input tensor of shape [batch_size * sequence_size, 3, 270, 480]
        :return: Output tensor of shape [9] if control_mode == "keyboard" or [2] if control_mode == "controller"
        """
            x = self.EncoderCNN(x)
            x = self.PositionalEncoding(x)
            x = self.EncoderTransformer(x)
            return self.OutputLayer(x)

class ModelPL(pl.LightningModule):
    """
    Pytorch Lightning module for the Tedd1104Model
    """

    def __init__(
        self,
        cnn_model_name: str,
        pretrained_cnn: bool,
        embedded_size: int,
        nhead: int,
        num_layers_encoder: int,
        lstm_hidden_size: int,
        dropout_cnn_out: float,
        positional_embeddings_dropout: float,
        dropout_encoder: float,
        mask_prob: float,
        dropout_encoder_features: float = 0.8,
        control_mode: str = "keyboard",
        sequence_size: int = 5,
        encoder_type: str = "transformer",
        bidirectional_lstm=True,
        weights: List[float] = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-3,
        label_smoothing: float = 0.0,
    ):
       

        super(ModelPL, self).__init__()

        self.encoder_type = encoder_type.lower()
        assert self.encoder_type in [
            "lstm",
            "transformer",
        ], f"Encoder type {self.encoder_type} not supported, supported feature encoders [lstm,transformer]."

        self.control_mode = control_mode.lower()

        assert self.control_mode in [
            "keyboard",
            "controller",
        ], f"{self.control_mode} control mode not supported. Supported dataset types: [keyboard, controller].  "

        self.cnn_model_name: str = cnn_model_name
        self.pretrained_cnn: bool = pretrained_cnn
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.nhead: int = nhead
        self.num_layers_encoder: int = num_layers_encoder
        self.dropout_cnn_out: float = dropout_cnn_out
        self.positional_embeddings_dropout: float = positional_embeddings_dropout
        self.dropout_encoder: float = dropout_encoder
        self.dropout_encoder_features = dropout_encoder_features
        self.mask_prob = mask_prob
        self.bidirectional_lstm = bidirectional_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.weights = weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing

        
        self.model = Transformer(
                embedded_size=self.embedded_size,
                nhead=self.nhead,
                num_layers_transformer=self.num_layers_encoder,
                dropout_cnn_out=self.dropout_cnn_out,
                positional_embeddings_dropout=self.positional_embeddings_dropout,
                dropout_transformer=self.dropout_encoder,
                mask_prob=self.mask_prob,
                control_mode=self.control_mode,
                sequence_size=self.sequence_size,
                dropout_encoder_features=self.dropout_encoder_features,
            )
        self.criterion = CrossEntropyLoss(
                weights=self.weights, label_smoothing=self.label_smoothing
            )
        self.Keyboard2Controller = Keyboard2Controller()

    def forward(self, x, output_mode: str = "keyboard", return_best: bool = True):
        """
        Forward pass of the model.

        :param x: input data [batch_size * sequence_size, 3, 270, 480]
        output_mode: output mode, either "keyboard" or "controller". 
        return_best: if True, we will return the class probabilities, else we will return the class with the highest probability (only for "keyboard" output_mode)
        """
        x = self.model(x)

        x = torch.functional.F.softmax(x, dim=1)
        if output_mode == "keyboard":
            if return_best:
                return torch.argmax(x, dim=1)
            else:
                return x

        elif output_mode == "controller":
            return self.Keyboard2Controller(x)
        else:
            raise ValueError(
                f"Output mode: {output_mode} not supported. Supported modes: [keyboard,controller]"
            )

if __name__ == "__main__":
    model = ModelPL.load_from_checkpoint(checkpoint_path = "epoch=11-step=456653.ckpt" )
    
