import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator ,Generator, initialize_weights
from utils import gradient_penalty

# when use gradient penalty with WGAN remove batchnorm2d and use instancenorm2d . use Lr=1e-4 . use lambada =10
# remove cliping as you areusing gradient penalty.  use ADAM insteat RMSprop. call gp=gradient_penalty() then use lossfunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Lr = 1e-4
batch_size = 64
image_size = 64
channels_img=1
z_dim = 100
num_epochs = 5
feature_disc = 64
feature_gen = 64
#weigth_clip = 0.01
critic_iteration = 5
lambada = 10
transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)]
        ),
    ]
)
datasets = datasets.MNIST(root="dataset/",train=True,transform=transforms,download=True)
#datasets = datasets.ImageFolder(root="celeb_dataset",transform =transforms)
loader = DataLoader(datasets, batch_size=batch_size,shuffle=True)

gen = Generator(z_dim,channels_img,feature_gen).to(device)
critic = Discriminator(channels_img,feature_disc).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(),lr = Lr,betas=(0.0,0.9))
opt_disc = optim.Adam(critic.parameters(),lr = Lr,betas=(0.0,0.9))


fixed_noise = torch.randn(32,z_dim,1,1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step=0

gen.train()
critic.train()

for epoch in range(num_epochs):
    for batch_idx,(real,_) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        for _ in range(critic_iteration):
            noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic,real,fake,device=device)
            loss_critic = (-(torch.mean(critic_real)-torch.mean(critic_fake)) +lambada *gp)
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            # for p in critic.parameters():
            #     p.data.clamp_(-weigth_clip,weigth_clip)
        # train generator min -E[critic(gen_fake)]
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        print(batch_idx," ",epoch)
        if batch_idx%100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise)

                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32],normalize=True)

                writer_fake.add_image(
                    "fake Images", img_grid_fake, global_step=step,
                )
                writer_real.add_image(
                    "real Images", img_grid_real, global_step=step,
                )
            step+=1
