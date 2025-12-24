import torch

from . import plots
import matplotlib.pyplot as plt

plots.update_mpl()

def plot_histogram(plot_loc,hist,xloc,epoch=1):
    
    fig = plt.figure(figsize=(15, 15))
    
    plt.stairs(hist,xloc)
    
    fig.savefig(plot_loc+f"/histogram_{epoch}.png")
    plt.close()

@torch.no_grad
def make_histogram(plot_loc,x,model,boundary):
    
    #plot_loc = self.logger.newpath 
    
    #make a plot of final points vs estimated distribution
    x_samples,_ = model.sample(5000)
    
    qaxtemp = torch.linspace(-5,5,1000,dtype=torch.float32).reshape((1000,1))
    pdf_temp = torch.exp(model.log_prob(qaxtemp))
    pdf_temp[torch.isnan(pdf_temp)] = 0

    fig2 = plt.figure(figsize=(15, 15))
    
    #samples from learned distribution
    plt.hist(x_samples.detach().numpy().flatten(),density=True,bins=100,alpha=0.5,color="orange")
    #plt.hist(x.detach().numpy().flatten(),density=True,bins=100,alpha=0.5)
    
    #pdf of learned distribution
    plt.plot(qaxtemp.detach().numpy().flatten(),pdf_temp.detach().numpy().flatten())    

    fig2.savefig(plot_loc+"/histogram.png")
    plt.close()


@torch.no_grad
def make_scatter_plot(plot_loc,x,model,epoch=1,boundary=None):
    
    #plot_loc = self.logger.newpath 
    
    #make a plot of final points vs estimated distribution
    x_samples,_ = model.sample(1000)

    #qsamples = boundary.generate_samples(1000,mode="initial")
    
    fig2 = plt.figure(figsize=(15, 15))
    
    #samples from distributions
    plt.scatter(x_samples[...,0].detach().numpy(),x_samples[...,1].detach().numpy(),color="blue")
    plt.scatter(x[...,0].detach().numpy(),x[...,1].detach().numpy(),color="orange")
    #plt.scatter(qsamples[...,0],qsamples[...,1])
    #plt.hist(x[...,0].detach().numpy(),density=True,bins=100,alpha=0.5,color="orange")
        
    fig2.savefig(plot_loc+f"/distribution_{epoch}.png")
    plt.close()
    
    #estimate kl divergence between the two sample sets
    #print("KL divergence between the samples",kl_div(x.detach().numpy(),x_samples.detach().numpy()))
    
@torch.no_grad
def distribution_plot(plot_loc,model,Tf,epoch=1,boundary=None):
    plotter = plots.PlotFuns(boundary)

    #compute marginals 
    xx, yy = torch.meshgrid(torch.linspace(-plotter.qmax,plotter.qmax, 100), torch.linspace(-plotter.qmax, plotter.qmax, 100), indexing="xy")
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    
    #find probabilities 
    pdf_temp = torch.exp(model.log_prob(zz))
    pdf_temp[torch.isnan(pdf_temp)] = 0
    #pdf_temp2 = torch.exp(model.log_prob(zz))
    #pdf_temp2[torch.isnan(pdf_temp2)] = 0

    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(25)

    xout,jointout = zz,pdf_temp.detach().clone().numpy().flatten()

    plotter.plot_joint_panel(plt.gca(),5,jointout,xout)

    fig.tight_layout()
    fig.savefig(plot_loc+f"/contour_distribution_{epoch}.png",bbox_inches="tight")        

    plt.close()

    fig1 = plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, pdf_temp.view(*xx.shape).detach().clone().numpy(), cmap='coolwarm')
    fig1.savefig(plot_loc+"/pdf_distribution.png",bbox_inches="tight") 
    plt.close()
