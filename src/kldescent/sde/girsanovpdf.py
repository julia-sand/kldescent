import torch

@torch.no_grad
def compute_log_pdf(x0, integrator, mc_samples=None, tsteps=None):
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

    tsteps : int, optional
        Number of time steps. Defaults to integrator.sde_params["tsteps"].

    Returns
    -------
    log_pdf : torch.Tensor
        Log-sum-exp estimate of log ρ(x0) via the Girsanov change of measure.
    """

    net       = integrator.net
    sigma0    = integrator.boundary.sigma0
    epsilon   = integrator.boundary.params["epsilon"]
    p_initial = integrator.boundary.p_initial
    Tf        = integrator.boundary.params["Tf"]
    tsteps    = tsteps     or integrator.sde_params["tsteps"]
    mc_samples = mc_samples or integrator.sde_params["testsamples"]

    net.eval()

    with torch.no_grad():

        if mc_samples == 1:
            xinit = x0.clone().to(net.device)
        else:
            xinit = x0.unsqueeze(0).expand(mc_samples, -1, -1).clone().to(net.device)

        t_vec = torch.linspace(0, Tf, tsteps, requires_grad=False, dtype=torch.float32).to(net.device)

        girs   = torch.zeros_like(xinit[...,[0]], requires_grad=False, dtype=torch.float32).to(net.device)
        pdfout = torch.zeros_like(xinit[...,[0]], requires_grad=False, dtype=torch.float32).to(net.device)

        for i in range(1, len(t_vec)):

            dt = torch.abs(t_vec[-i-1] - t_vec[-i])

            dw = integrator.generate_innovation(dt, xinit[...,[0]])

            du_bar = net(torch.cat((t_vec[-i]*torch.ones_like(xinit[...,[0]], dtype=torch.float32).to(net.device),
                                    xinit[...,[0]]), dim=-1))

            xinit = integrator.girsanov_step(xinit, du_bar, dt, dw)

            girs = girs + (dw*xinit[...,[1]])/(2**0.5) + (dt * (xinit[...,[1]])**2)/4

        log_p = torch.log(p_initial(xinit[...,[0]], xinit[...,[1]]))
        log_w = log_p - girs

    if log_w.dim() > 2:
        return torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(float(mc_samples)))
    else:
        return log_w

def plot_scatter_(x,pdf):
    plt.scatter(x[...,0].cpu().numpy(), x[...,1].cpu().numpy(), c=pdf.cpu().numpy(),cmap='RdYlGn_r')
    plt.title(f"{pdf.cpu().numpy().max()}")
    randint = random.randint(1, 100000) 
    plt.savefig(f"test_tests1.png")
    plt.close()

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import random
    from kldescent.sde.boundary import UnderdampedLandauer
    from kldescent.model.nn import NeuralNetwork

    boundary = UnderdampedLandauer()
    nn_temp = NeuralNetwork(boundary)
    testing_samples = 1000
    with torch.no_grad():
        xf = boundary.generate_samples(testing_samples,mode="final")

        log_pdf = compute_log_pdf(xf, nn_temp, mc_samples=2, tsteps=21)

    plot_scatter_(xf, torch.exp(log_pdf))

