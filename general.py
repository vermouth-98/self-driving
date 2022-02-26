from torch import nn
import torch
class Keyboard2Controller(nn.Module):
    """
    Map keyboard keys probabilities to controller output
    """

    def __init__(self):
        """
        INIT
        """
        super(Keyboard2Controller, self).__init__()
        keys2vector_matrix = torch.tensor(
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            requires_grad=False,
        )

        self.register_buffer("keys2vector_matrix", keys2vector_matrix)

    def forward(self, x: torch.tensor):
        """
        Forward pass

        :param torch.tensor x: Keyboard keys probabilities [9]
        :return: Controller input [2]
        """
        controller_inputs = self.keys2vector_matrix.repeat(len(x), 1, 1)
        return (
            torch.sum(controller_inputs * x.view(len(x), 9, 1), dim=1)
            / torch.sum(x, dim=-1)[:, None]
        )