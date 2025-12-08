import torch

from . import plots
import matplotlib.pyplot as plt



plots.update_mpl()

plotter = plots.PlotFuns()

@torch.no_grad
def make_scatter_plot(plot_loc,x,model):
    
    #plot_loc = self.logger.newpath 
    
    #make a plot of final points vs estimated distribution
    x_samples,_ = model.sample(1000)
    
    fig2 = plt.figure(figsize=(15, 15))
    #samples from distributions
    plt.scatter(x_samples[...,0].detach().numpy(),x_samples[...,1].detach().numpy(),color="blue")
    plt.scatter(x[...,0].detach().numpy(),x[...,1].detach().numpy(),color="orange")
    #plt.hist(x[...,0].detach().numpy(),density=True,bins=100,alpha=0.5,color="orange")
        
    fig2.savefig(plot_loc+"/distribution.png")
    plt.close()
    
    #estimate kl divergence between the two sample sets
    #print("KL divergence between the samples",kl_div(x.detach().numpy(),x_samples.detach().numpy()))
    
@torch.no_grad
def distribution_plot(plot_loc,model,Tf):

    #compute marginals 
    xx, yy = torch.meshgrid(torch.linspace(-plotter.qmax,plotter.qmax, 10), torch.linspace(-plotter.qmax, plotter.qmax, 10), indexing="ij")
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    
    #find probabilities 
    pdf_temp = torch.exp(model.log_prob(zz))
    pdf_temp[torch.isnan(pdf_temp)] = 0
    pdf_temp2 = torch.exp(model.log_prob(zz).view(*xx.shape))
    pdf_temp2[torch.isnan(pdf_temp2)] = 0

    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(25)


    xout,jointout = zz,pdf_temp.data.numpy().flatten()
    print(zz.shape)
    plotter.plot_joint_panel(plt.gca(),5,jointout,xout)

    fig.tight_layout()
    fig.savefig(plot_loc+"/contour_distribution.png",bbox_inches="tight")        

    plt.close()

    fig1 = plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, pdf_temp2.data.numpy(), cmap='coolwarm')
    fig1.savefig(plot_loc+"/pdf_distribution.png",bbox_inches="tight") 
    plt.close()
