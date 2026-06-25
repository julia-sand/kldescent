"""Loss function wrapper decoupled from the integrator."""

import torch


class KL:
    """
    Compute the KL-divergence loss (Proposition 3, arXiv:2410.05163) using
    a pre-constructed integrator for the SDE stepping.

    Parameters
    ----------
    integrator : SDEBase subclass instance
        An already-initialised integrator (EulerMaruyamaSDE, BAOAB, VerletSDE,
        …). Must expose: net, boundary, sde_params, device,
        integration_step(), generate_innovation(), stratified_time_grid().
    terminalcost : object
        Terminal cost handler; must expose gfun(xinit, integrator, n_samples).
    """

    def __init__(self, integrator, terminalcost):
        
        self.integrator = integrator
        self.net = integrator.net
        self.boundary = integrator.boundary
        self.sde_params = integrator.sde_params
        self.device = integrator.device
        
        self.TerminalCost = terminalcost

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def __call__(self):
        return self.compute()

    def compute(self):
        """
        Roll out trajectories and return the scalar loss Lhat.

        Compute 
        a = ∫‖du‖²dt,
        and 
        c = ∫du·dw 
        along trajectories. 
        
        The loss is computed as:
        L̂ = mean(0.5a + (0.5abar + g)·c) 

        with g the terminal cost computed from the terminal cost class.

        Returns
        -------
        Lhat : torch.Tensor (scalar)
        """
        self.net.train()

        t_vec = self.integrator.stratified_time_grid()
        n = self.sde_params["batchsamples"]

        xinit = self.boundary.generate_samples(n, "initial").to(self.device)

        a    = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        abar = torch.zeros((n, 1), dtype=torch.float32, device=self.device,
                           requires_grad=False)
        #bbar = torch.zeros((n, 1), dtype=torch.float32, device=self.device,
        #                   requires_grad=False)
        c    = torch.zeros((n, 1), dtype=torch.float32, device=self.device)

        for i in range(len(t_vec) - 1):
            dt = t_vec[i + 1] - t_vec[i]

            dw = self.integrator.generate_innovation(dt, xinit[..., :self.boundary.d])

            du = self.net(
                torch.cat((
                    t_vec[i] * torch.ones((n, 1), dtype=torch.float32,
                                          device=self.device),
                    xinit[..., :self.boundary.d]
                ), dim=-1)
            )

            a += dt * (torch.linalg.norm(du, dim=-1, keepdim=True) ** 2)
            c += (du*dw).sum(dim=-1, keepdim=True)

            du_bar = du.detach()

            with torch.no_grad():

                abar += dt * (
                    torch.linalg.norm(du_bar, dim=-1, keepdim=True) ** 2
                )

                #make final half-step for BAOAB integrators
                #skip for other methods and when t=0
                #xinit = self.integrator.loss_intermediate_step(xinit,dt, t_vec[i], du_bar)

                xinit = self.integrator.loss_integration_step(
                    xinit, dt, t_vec[i], du_bar, dw
                )

        with torch.no_grad():

            xinit = self.integrator.loss_final_step(xinit,t_vec)
            gvals = self.TerminalCost.gfun(
                xinit.detach().clone(),
                self.integrator,
                self.sde_params["testsamples"],
            ).to(self.device)

            

        Lhat = torch.mean(0.5 * a + (0.5 * abar + gvals) * c)
        return Lhat

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

class ep:
    """
    Placeholder for the entropy production loss formulation.

    Parameters
    ----------
    integrator : SDEBase subclass instance
        An already-initialised integrator (EulerMaruyamaSDE, BAOAB, VerletSDE,
        …). Must expose: net, boundary, sde_params, device,
        integration_step(), generate_innovation(), stratified_time_grid().
    terminalcost : object
        Terminal cost handler; must expose gfun(xinit, integrator, n_samples).
    """

    def __init__(self, integrator, terminalcost):
        self.integrator = integrator
        self.net = integrator.net
        self.boundary = integrator.boundary
        self.sde_params = integrator.sde_params
        self.device = integrator.device
        self.TerminalCost = terminalcost

    def __call__(self):
        return self.compute()

    def compute(self):
        raise NotImplementedError
