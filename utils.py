import torch
import numpy as np
import matplotlib.pyplot as plt

def gradient_penalty(f, real_data, fake_data):
    alpha = torch.rand(real_data.shape[0], 1, 1, 1)
    alpha = alpha.expand(real_data.size()).cuda()

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).detach()
    interpolates.requires_grad = True

    disc_interpolates = f(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    return ((gradients.reshape(real_data.shape[0],-1).norm(2, dim=1) - 1) ** 2).mean()

def print_images(x, ax, n=7, d=32, color='black'):
    ax.cla()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    panel = np.zeros([x.shape[1],n*d,n*d])
    for i in range(n):
        for j in range(n):
            panel[:,i*d:(i+1)*d,j*d:(j+1)*d] = x[i*n+j]

    ax.imshow(panel.transpose(1,2,0).squeeze(), cmap=plt.get_cmap('Greys'))
    plt.setp(ax.spines.values(), color=color)

