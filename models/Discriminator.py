from torch import nn


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args.pred_len, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


class Generator_Loss(nn.Module):
    def __init__(self, args):
        super(Generator_Loss, self).__init__()
        self.loss = nn.MSELoss()
        self.adv_loss = nn.BCELoss()

    def forward(self, pred, target):
        loss = self.loss(pred, target) + 0.1 * self.adv_loss(discriminator(pred), target)
        return loss
