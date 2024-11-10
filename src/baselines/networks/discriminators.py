from torch import nn
from src.utils import deterministic_NeuralSort

from src.evaluation.strategies import *
equal_weight = EqualWeightPortfolioStrategy()
mean_reversion = MeanReversionStrategy()
trend_following = TrendFollowingStrategy()
volatility_trading = VolatilityTradingStrategy()

R_shape = (3,)
class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, out_dim=1):
        super(LSTMDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_size=input_dim+1,
                            hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, condition.unsqueeze(1).repeat((1, x.shape[1], 1))], dim=2)
        h = self.lstm(z)[0][:, -1:]
        x = self.linear(h)
        return x

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.W = config.W
        self.project = config.project
        self.alphas = config.alphas
        # self.model = nn.Sequential(
        #     nn.Linear(3, 256), # nn.Linear(3, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128, 2 * len(config.alphas)),
        # )
        self.model_pnl = nn.Sequential(
            # nn.Linear(13, 256), # nn.Linear(3, 256),
            # nn.Linear(38, 256),
            nn.Linear(64, 512), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(128, 2 * len(config.alphas)),
            nn.Linear(256, 4 * 2 * len(config.alphas)),
        )

    def project_op(self, validity):
        for i, alpha in enumerate(self.alphas):
            v = validity[:, 2*i].clone()
            e = validity[:, 2*i+1].clone()
            indicator = torch.sign(torch.as_tensor(0.5 - alpha))
            validity[:, 2*i] = indicator * ((self.W * v < e).float() * v + (self.W * v >= e).float() * (v + self.W * e) / (1 + self.W ** 2))
            validity[:, 2*i+1] = indicator * ((self.W * v < e).float() * e + (self.W * v >= e).float() * self.W * (v + self.W * e) / (1 + self.W ** 2))
        return validity


    def forward(self, x):
        PNL = mean_reversion.get_pnl_trajectory(x)
        PNL_s = PNL.reshape(*PNL.shape, 1).to(self.config.device)
        perm_matrix = deterministic_NeuralSort(PNL_s, self.config.temp)
        PNL_sort = torch.bmm(perm_matrix, PNL_s)
        batch_size, seq_len, _ = PNL_s.shape

        PNL2 = equal_weight.get_pnl_trajectory(x)
        PNL2_s = PNL2.reshape(*PNL2.shape, 1).to(self.config.device)
        perm_matrix2 = deterministic_NeuralSort(PNL2_s, self.config.temp)
        PNL_sort2 = torch.bmm(perm_matrix2, PNL2_s)


        PNL3 = trend_following.get_pnl_trajectory(x)
        PNL3_s = PNL3.reshape(*PNL3.shape, 1).to(self.config.device)
        perm_matrix3 = deterministic_NeuralSort(PNL3_s, self.config.temp)
        PNL_sort3 = torch.bmm(perm_matrix3, PNL3_s)

        PNL4 = volatility_trading.get_pnl_trajectory(x)
        PNL4_s = PNL4.reshape(*PNL4.shape, 1).to(self.config.device)
        perm_matrix4 = deterministic_NeuralSort(PNL4_s, self.config.temp)
        PNL_sort4 = torch.bmm(perm_matrix4, PNL4_s)


        PNL_cat = torch.cat((PNL, PNL2, PNL3, PNL4), dim=1)
        PNL_sort_cat = torch.cat((PNL_sort, PNL_sort2, PNL_sort3, PNL_sort4), dim=1)
        PNL_validity = self.model_pnl(PNL_sort_cat.view(batch_size, -1))

        # PNL_cat = torch.cat((PNL, PNL2), dim=1)
        # PNL_sort_cat = torch.cat((PNL_sort, PNL_sort2), dim=1)
        # PNL_validity = self.model_pnl(PNL_sort_cat.view(batch_size, -1))

        # PNL_validity = self.model_pnl(PNL_sort.view(batch_size, -1))
        # return PNL, PNL_validity
    
        return PNL_cat, PNL_validity
    
    
