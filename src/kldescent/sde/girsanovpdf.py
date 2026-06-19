import torch

@torch.no_grad
def compute_log_pdf(x0, integrator, mc_samples=None, Tf=None):
    """
    Compute an estimate of the final joint distribution using Girsanov's theorem.

    Gradient tracking is disabled.

    Based on: arXiv:2411.08518 (see Section 3 for an example application).

    Parameters
    ----------
    x0 : torch.Tensor
        Coordinates (q, p) of x₀ at which to evaluate the density rho.

    integrator : SDEBase subclass instance
        Initialised integrator; provides net, boundary, sde_params, device,
        and generate_innovation().

    mc_samples : int, optional
        Number of Monte Carlo samples. Defaults to integrator.sde_params["testsamples"].

    Tf : float, optional
        Final time. Defaults to integrator.boundary.params["Tf"].

    Returns
    -------
    log_pdf : torch.Tensor
        Log-sum-exp estimate of log ρ(x0) via the Girsanov change of measure.
    """

    device       = integrator.net.device
    
    if Tf is None:
        Tf = integrator.boundary.params["Tf"]
    
    tsteps = int(integrator.sde_params["tsteps"]*(Tf/integrator.boundary.params["Tf"]))

    mc_samples = mc_samples or integrator.sde_params["testsamples"]

    integrator.net.eval()

    with torch.no_grad():

        if mc_samples == 1:
            xinit = x0.clone().to(device)
        else:
            xinit = x0.unsqueeze(0).expand(mc_samples, -1, -1).clone().to(device)

        t_vec = integrator.stratified_time_grid(Tf, tsteps)
        #torch.linspace(0, Tf, tsteps, requires_grad=False, dtype=torch.float32).to(device)

        girs   = torch.zeros_like(xinit[...,[0]], requires_grad=False, dtype=torch.float32)
        #dw = torch.zeros_like(xinit[...,[0]], requires_grad=False, dtype=torch.float32).to(device)

        for i in range(1, len(t_vec)):

            dt = t_vec[-i]-t_vec[-i-1]

            dw = integrator.generate_girsanov_innovation(dt, xinit[...,[0]])

            du_bar = integrator.net(torch.cat((t_vec[-i]*torch.ones_like(xinit[...,[0]], dtype=torch.float32).to(device),
                                    xinit[...,[0]]), dim=-1))

            xinit = integrator.girsanov_step(xinit, du_bar, dt, dw)

            girs += (dw*xinit[...,[1]])/(2**0.5) + (dt * (xinit[...,[1]])**2)/4

        log_p = integrator.boundary.log_pinitial(xinit[...,[0]], xinit[...,[1]])
        #torch.log(p_initial(xinit[...,[0]], xinit[...,[1]]))
        log_w = -log_p - girs

    if log_w.dim() > 2:
        return torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(float(mc_samples)))
    else:
        return log_w

def plot_scatter_(x,pdf):
    plt.scatter(x[...,0].cpu().numpy(), x[...,1].cpu().numpy(), c=pdf.cpu().numpy(),cmap='RdYlGn_r')
    plt.title(f"{pdf.cpu().numpy().max()}")
    #randint = random.randint(1, 100000) 
    plt.savefig(f"test_tests1.png")
    plt.close()

if __name__=="__main__":
    import matplotlib.pyplot as plt
    #import random
    from kldescent.sde.boundary import UnderdampedLandauer
    from kldescent.model.nn import NeuralNetwork

    boundary = UnderdampedLandauer()
    nn_temp = NeuralNetwork(boundary)
    testing_samples = 1000
    with torch.no_grad():
        xf = boundary.generate_samples(testing_samples,mode="final")

        log_pdf = compute_log_pdf(xf, nn_temp, mc_samples=2)

    plot_scatter_(xf, torch.exp(log_pdf))

