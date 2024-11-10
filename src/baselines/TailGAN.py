from torch import nn
from src.baselines.base import BaseTrainer
from tqdm import tqdm
from os import path as pt
from src.utils import save_obj
from src.utils import load_config
from src.evaluation.strategies import *
import matplotlib.pyplot as plt

from src.evaluation.loss import CumulativePnLLoss, MaxDrawbackLoss
import numpy as np

config_dir = pt.join("configs/config.yaml")
config = (load_config(config_dir))


def G1(v):
    return v

def G2(e, scale=1):
    return scale * torch.exp(e / scale)

def G2in(e, scale=1):
    return scale ** 2 * torch.exp(e / scale)

def G1_quant(v, W=config.W):
    return - W * v ** 2 / 2

def G2_quant(e, alpha):
    return alpha * e

def G2in_quant(e, alpha):
    return alpha * e ** 2 / 2

def S_stats(v, e, X, alpha):
    """
    For a given quantile, here named alpha, calculate the score function value
    """
    if alpha < 0.5:
        rt = ((X<=v).float() - alpha) * (G1(v) - G1(X)) + 1. / alpha * G2(e) * (X<=v).float() * (v - X) + G2(e) * (e - v) - G2in(e)
    else:
        alpha_inverse = 1 - alpha
        rt = ((X>=v).float() - alpha_inverse) * (G1(X) - G1(v)) + 1. / alpha_inverse * G2(-e) * (X>=v).float() * (X - v) + G2(-e) * (v - e) - G2in(-e)
    return torch.mean(rt)

def S_quant(v, e, X, alpha, W=config.W):
    """
    For a given quantile, here named alpha, calculate the score function value
    """
    # print(X.device) #CPU
    # print(v.device) #CUDA
    X = X.to(v.device)
    if alpha < 0.5:
        rt = ((X<=v).float() - alpha) * (G1_quant(v,W) - G1_quant(X,W)) + 1. / alpha * G2_quant(e,alpha) * (X<=v).float() * (v - X) + G2_quant(e,alpha) * (e - v) - G2in_quant(e,alpha)
    else:
        alpha_inverse = 1 - alpha
        rt = ((X>=v).float() - alpha_inverse) * (G1_quant(v,W) - G1_quant(X,W)) + 1. / alpha_inverse * G2_quant(-e,alpha_inverse) * (X>=v).float() * (X - v) + G2_quant(-e,alpha_inverse) * (v - e) - G2in_quant(-e,alpha_inverse)
        
    return torch.mean(rt)


class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.alphas = config.alphas
        self.score_name = config.score
        if self.score_name == 'quant':
            self.score_alpha = S_quant
        elif self.score_name == 'stats':
            self.score_alpha = S_stats
        else:
            self.score_alpha = None

    def forward(self, PNL_validity, PNL):
        # Score
        loss = 0
        
        # for i, alpha in enumerate(self.alphas):
        #     PNL_var = PNL_validity[:, [(2 * i)]]
        #     PNL_es = PNL_validity[:, [2 * i + 1]]
        #     loss += self.score_alpha(PNL_var, PNL_es, PNL, alpha)

        strat_num = 4
        strat_days_list = [13,38,51,64]

        for i, alpha in enumerate(self.alphas):
            for j, strat_days in enumerate(strat_days_list):
                PNL_var = PNL_validity[:, [(strat_num * 2 * i) + (2 * j)]]
                PNL_es = PNL_validity[:, [( (strat_num * 2 * i) + (2 * j) ) + 1]]
                if strat_days == 13:
                    loss += (self.score_alpha(PNL_var, PNL_es, PNL[:, :strat_days_list[j]], alpha) / strat_num)
                else:
                    loss += (self.score_alpha(PNL_var, PNL_es, PNL[:, strat_days_list[j-1] : strat_days_list[j]], alpha) / strat_num)
        # print("score loss", loss)
        return loss

class Score2(nn.Module):
    def __init__(self):
        super(Score2, self).__init__()

    def forward(self, PNL_Real, PNL_Fake):
        # print(PNL_Real.device) #CPU
        # print(PNL_Fake.device) #CPU
        # PNL_Real = PNL_Real.to(PNL_Fake.device)

        loss_G2 = 0
        loss_G3 = 0
        strat_num = 4
        strat_days_list = [13,38,51,64]

        for j, strat_days in enumerate(strat_days_list):

            # loss_G2
            if strat_days == 13:
                
                curr_PNL = PNL_Real[:, :strat_days_list[j]]
                curr_gen_PNL = PNL_Fake[:, :strat_days_list[j]]

                running_max_PNL = torch.cummax(curr_PNL, dim=1)[0]
                drawdowns_PNL = (running_max_PNL - curr_PNL)
                max_drawdown_PNL = torch.max(drawdowns_PNL, dim=1)[0]

                running_max_gen_PNL = torch.cummax(curr_gen_PNL, dim=1)[0]
                drawdowns_gen_PNL = (running_max_gen_PNL - curr_gen_PNL)
                max_drawdown_gen_PNL = torch.max(drawdowns_gen_PNL, dim=1)[0]

                loss_G2 += torch.mean(torch.abs(max_drawdown_PNL - max_drawdown_gen_PNL))/strat_num
            else:
                
                curr_PNL = PNL_Real[:, strat_days_list[j-1] : strat_days_list[j]]
                curr_gen_PNL = PNL_Fake[:, strat_days_list[j-1] : strat_days_list[j]]

                running_max_PNL = torch.cummax(curr_PNL, dim=1)[0]
                drawdowns_PNL = (running_max_PNL - curr_PNL)
                max_drawdown_PNL = torch.max(drawdowns_PNL, dim=1)[0]

                running_max_gen_PNL = torch.cummax(curr_gen_PNL, dim=1)[0]
                drawdowns_gen_PNL = (running_max_gen_PNL - curr_gen_PNL)
                max_drawdown_gen_PNL = torch.max(drawdowns_gen_PNL, dim=1)[0]

                loss_G2 += torch.mean(torch.abs(max_drawdown_PNL - max_drawdown_gen_PNL))/strat_num
            
            
            # loss_G3
            loss_G3 += torch.abs(torch.mean(PNL_Real[:,strat_days-1] - PNL_Fake[:,strat_days-1]))/strat_num

        loss = loss_G2 + loss_G3
        return loss



class TailGANTrainer(BaseTrainer):
    def __init__(self, D, G, train_dl, config,
                 **kwargs):
        super(TailGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)),
            **kwargs
        )

        self.config = config
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.D = D
        self.D_optimizer = torch.optim.Adam(
            D.parameters(), lr=config.lr_D, betas=(0, 0.9))  # TailGAN: lr=1e-7, betas=(0.5, 0.999)

        self.train_dl = train_dl
        self.reg_param = 0
        self.criterion = Score()
        # self.criterion2 = Score2()

    def fit(self, device):
        self.G.to(device)
        self.D.to(device)

        ##########################################################################################
        print("initial discriminator training")
        for i in tqdm(range(25)):
            # generate x_fake

            with torch.no_grad():
                x_real_batch = next(iter(self.train_dl)) # .to(device) # init_price & log_return
                x_fake_log_return = self.G(
                    batch_size=self.batch_size,
                    n_lags=24,
                    device=device,
                )

            x_fake = [x_real_batch[0], x_fake_log_return] 
            init_prices_real = x_real_batch[0] 
            log_returns_real = x_real_batch[1]
            price_real = log_return_to_price(log_returns_real, init_prices_real)
            init_prices_gen = x_fake[0] 
            log_returns_gen = x_fake[1]
            price_gen = log_return_to_price(log_returns_gen, init_prices_gen)
            D_loss = self.D_trainstep(price_gen, price_real)
            if i == 0:
                self.losses_history["D_loss"].append(D_loss)
        D_loss_copy = torch.tensor(self.losses_history["D_loss"], device = 'cpu')
        plt.plot(D_loss_copy, label='D') # Added
        plt.legend(loc="upper left")
        plt.show() # Added
        print("fine-tuning loop")
        ##########################################################################################
        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)
            # if i % 1 == 0:
        D_loss_copy = torch.tensor(self.losses_history["D_loss"], device = 'cpu')
        # G1_loss_copy = torch.tensor(self.losses_history["G1_loss"], device = 'cpu')
        # G2_loss_copy = torch.tensor(self.losses_history["G2_loss"], device = 'cpu')
        plt.plot(D_loss_copy, label='D') # Added
        # plt.plot(G1_loss_copy, label='G1') # Added
        # plt.plot(G2_loss_copy, label='G2') # Added
        plt.legend(loc="upper left")
        plt.show() # Added

    
    def step(self, device, step):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake

            with torch.no_grad():
                x_real_batch = next(iter(self.train_dl)) # .to(device) # init_price & log_return
                # x_fake_log_return = self.G(self.config.batch_size, device)
                ####################################
                ######  DELETE #####################
                ####################################
                x_fake_log_return = self.G(
                    batch_size=self.batch_size,
                    n_lags=24,
                    device=device,
                )
                ####################################

            x_fake = [x_real_batch[0], x_fake_log_return] 
            init_prices_real = x_real_batch[0] 
            log_returns_real = x_real_batch[1]
            price_real = log_return_to_price(log_returns_real, init_prices_real)
            init_prices_gen = x_fake[0] 
            log_returns_gen = x_fake[1]
            price_gen = log_return_to_price(log_returns_gen, init_prices_gen)
            D_loss = self.D_trainstep(price_gen, price_real)
            if i == 0:
                self.losses_history["D_loss"].append(D_loss)
        G_loss = self.G_trainstep(price_gen, price_real, device, step)
        # self.losses_history["G_loss"].append(G_loss)
        # self.losses_history["G_loss"].append(G_loss) # Added
        # print (f'Epoch [{step+1}/{self.n_gradient_steps}], G_Loss: [{(G_loss):.4f}], D_Loss: [{(D_loss):.4f}]') # Added

        ################### testing

        # x_fake_log_return2 = self.G(self.config.batch_size, device)

        # equal_weight = EqualWeightPortfolioStrategy()
        # mean_reversion = MeanReversionStrategy()
        # trend_following = TrendFollowingStrategy()
        # volatility_trading = VolatilityTradingStrategy()
        # self.G.train()
        # self.G_optimizer.zero_grad()
        # loss_test = 0
        # for i in [equal_weight]:
        #     # test1 = torch.exp(x_fake_log_return2)
        #     # test2 = torch.cumprod(test1, dim=1)
        #     # test_init = x_real_batch[0].clone()
        #     # test_init2 = test_init.repeat(1, test2.shape[1], 1)
        #     # test3 = test2 * test_init2
        #     # price_gen = log_return_to_price(x_fake_log_return2, x_real_batch[0])

        #     # PNL_test = trend_following.get_pnl_trajectory(price_gen)
        #     price_gen = log_return_to_price(x_fake_log_return2, x_real_batch[0])
        #     PNL_gen_test = i.get_pnl_trajectory(price_gen)
        #     # PNL_test = i.get_pnl_trajectory(price_real)
        #     # loss_test = torch.abs(torch.mean(PNL_Real[:,-1] - PNL_Fake[:,-1]))
        #     loss_test += torch.abs(torch.mean(PNL_gen_test[:,-1] - PNL_gen_test[:,-1]))/4
        #     # loss_test = torch.mean(PNL_test[:,-1])

        # loss_test.backward(retain_graph=True) # Corrected
        # self.G_optimizer.step()

            

    def D_trainstep(self, x_fake, x_real):
        self.D_optimizer.zero_grad()
        # Adversarial loss
        self.D = self.D.to(config.device)
        PNL, PNL_validity = self.D(x_real)
        gen_PNL, gen_PNL_validity = self.D(x_fake)
        real_score = self.criterion(PNL_validity, PNL)   
        fake_score = self.criterion(gen_PNL_validity, PNL)
        # loss_D = real_score - fake_score #testing below
        loss_D = torch.abs(real_score - fake_score)
        # Update the Gradient in Discriminator
        loss_D.backward(retain_graph=True) # Corrected
        # loss_D.backward()
        self.D_optimizer.step()
        return loss_D.item()

    def G_trainstep(self, x_fake, x_real, device, step):
        PNL, PNL_validity = self.D(x_real)
        self.G.train()
        self.G_optimizer.zero_grad()
        self.D.train()
        gen_PNL, gen_PNL_validity = self.D(x_fake)
        loss_G = self.criterion(gen_PNL_validity, PNL)

        ######################################
        # loss_G1 = self.criterion(gen_PNL_validity, PNL)
        # loss_G2 = self.criterion2(PNL, gen_PNL)
        # self.losses_history["G1_loss"].append(loss_G1)
        # self.losses_history["G2_loss"].append(loss_G2)
        #####################################

        # loss_G = loss_G1 + loss_G2
        # loss_G = loss_G1
        # Update the Gradient in Generator
        loss_G.backward(retain_graph=True) # Corrected
        # loss_G.backward()
        self.G_optimizer.step()
        # print("test item", loss_G.item()) # ADDED
        return loss_G.item()
    
    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = torch.nn.BCELoss()(torch.nn.Sigmoid()(d_out), targets)
        return loss

    def save_model_dict(self):
        save_obj(self.G.state_dict(), pt.join(
            self.config.exp_dir, 'generator_state_dict.pt'))
