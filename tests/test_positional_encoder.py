import torch

from nerf_app.models.PositionalEncoder import PositionalEncoder


def test_positional_encoder():
    """
    Test Positional Encoder
    """

    encoder = PositionalEncoder(3, 5)
    input_tensorf = torch.rand((10000, 3))
    print(f"input_tensorf: {input_tensorf.shape}")

    encoder_rez = encoder(input_tensorf)

    print("Encoded Points")
    print(encoder_rez.shape)
    assert encoder_rez.shape[1] == input_tensorf.shape[1]* (1 + 2*5)
    print(torch.min(encoder_rez), torch.max(encoder_rez), torch.mean(encoder_rez))
    print("-" * 80)
