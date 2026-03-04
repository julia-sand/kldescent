
@torch.no_grad()
def generate_innovation(self,dt,size_like):
    """generate a step of brownian motion"""
    return np.sqrt(dt)*torch.randn_like(size_like,requires_grad=False,dtype=torch.float32)

    def loss_fn(self):
        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters

        ====
        """
        
        #self.net.train()
        t_vec = self.generate_time_grid()

        #initialise x values 
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),requires_grad=False,dtype=torch.float32)#.to_device(device)
        
        #set up a,abar,c,cbar
        a = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        abar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)
        bbar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)
        c = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        dw = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)

        for i in range(0,self.sde_params["tsteps"]-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = self.generate_innovation(dt,xinit[...,[0]])
           
            #get neural network drift
            du = self.net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32)
                                                    ,xinit[...,[0]])))
                    
            a = a + dt*(torch.linalg.norm(du, axis=1, keepdim=True) ** 2) 
            
            c = c + (du*dw)
            du_bar = du.detach() #copy without gradient 

            with torch.no_grad(): #no gradient 

                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True) ** 2) 
                bbar = bbar + dt*(self.boundary.f(xinit,t_vec[i])) 
                
                xinit = self.boundary.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])

        with torch.no_grad():
            #compute terminal cost
            gvals = self.TerminalCost.gfun(xinit.clone().detach())
                
        #compute value of Lhat
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + gvals) * c))
        
        return Lhat
    