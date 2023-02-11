import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class GPR(torch.nn.Module):
    """
    Gaussian Process Regression Model For Federated Learning
    This is a base class and a Covariance() function must be implemented by its subclasses

    Parameters:     
        Define by the kernel/covariance matrix
    
    Non-Parameter Tensor:
        noise: Standard Deviation of sample noise (sigma_n). 
               This noise is only to avoid singular covariance matrix.
        mu: Mean Priori, which is fixed while training. It can be evaluated by MLE with weighted data
    """
    def __init__(self,num_users,loss_type = 'LOO',init_noise = 0.01,reusable_history_length=10,gamma=1.0,device = torch.device('cpu')):
        """
        Arguments:
            num_users: Number of users in a Federated Learning setting
            loss_type: {"LOO","MML"}
        """
        super(GPR, self).__init__()
        self.num_users = num_users
        self.loss_type = loss_type
        # sigma_n
        self.noise = torch.tensor(init_noise,device=device)
        self.mu = torch.zeros(num_users,device=device).detach()
        self.discount = torch.ones(num_users,device=device).detach()
        self.data = {}
        self.reusable_history_length = reusable_history_length
        self.gamma = gamma
        self.device = device

    def Covariance(self,ids = None):
        raise NotImplementedError("A GPR class must have a function to calculate covariance matrix")

    def Update_Training_Data(self,client_idxs,loss_changes,epoch):
        """
        The training data should be in the form of : data[epoch] = sample_num x [user_indices, loss_change] (N x 2)
        Thus the data[epoch] is in shape S x N x 2
        """
        data = np.concatenate([np.expand_dims(np.array(client_idxs),2),np.expand_dims(np.array(loss_changes),2)],2)
        self.data[epoch] = torch.tensor(data,device=self.device,dtype=torch.float)
        for e in list(self.data.keys()):
            if e+self.reusable_history_length<epoch:
                self.data.pop(e) # delete too old data


    def Posteriori(self,data):
        """
        Get the posteriori with the data
        data: given in the form [index,loss]
        return:mu|data,Sigma|data
        """
        data = torch.tensor(data,device=self.device,dtype=torch.float)
        indices = data[:,0].long()
        values = data[:,1]
        
        Cov = self.Covariance()
        
        Sigma_inv = torch.inverse(Cov[indices,:][:,indices])
        mu = self.mu+((Cov[:,indices].mm(Sigma_inv)).mm((values-self.mu[indices]).unsqueeze(1))).squeeze()
        Sigma = Cov-(Cov[:,indices].mm(Sigma_inv)).mm(Cov[indices,:])
        return mu.detach(),Sigma.detach()


    def Log_Marginal_Likelihood(self,data):
        """
        MML:
        Calculate the log marginal likelihood of the given data
        data: given in the form S x [index,loss]
        return log(p(loss|mu,sigma,correlation,sigma_n))
        """
        res = 0.0

        for d in data:
            idx = d[:,0].long()
            val = d[:,1]
            Sigma = self.Covariance(idx)
            distribution = MultivariateNormal(loc = self.mu[idx],covariance_matrix = Sigma)
            res += distribution.log_prob(val)

        return res

    def Log_LOO_Predictive_Probability(self,data):
        """
        LOO:
        Calculate the Log Leave-One-Out Predictive Probability of the given data
        data: given in the form S x [index,loss]
        return: \sum log(p(y_i|y_{-i},mu,sigma,relation,sigma_n))
        """

        # High efficient algorithm exploiting partitioning
        log_p = 0.0
        for d in data:
            idx = d[:,0].long()
            val = d[:,1]
            Sigma_inv = torch.inverse(self.Covariance(idx))
            K_inv_y = (Sigma_inv.mm((val-self.mu[idx]).unsqueeze(1))).squeeze()
            for i in range(len(data)):
                mu = val[i]-K_inv_y[i]/Sigma_inv[i,i]
                sigma = torch.sqrt(1/Sigma_inv[i,i])
                dist = Normal(loc = mu,scale = sigma)
                log_p+=dist.log_prob(val[i])
        
        return log_p

    def Parameter_Groups(self):
        raise NotImplementedError("A GPR class must have a function to get parameter groups = [Mpar,Spar]")
    
    def MLE_Mean(self):
        """
        Calculate the weighted mean of historical data
        """
        self.mu = torch.zeros(self.num_users,device=self.device).detach()
        current_epoch = max(self.data.keys())
        cum_gamma = torch.zeros(self.num_users,device=self.device)
        for e in self.data.keys():
            for d in self.data[e]:
                idx = d[:,0].long()
                val = d[:,1]
                self.mu[idx] += (self.gamma**(current_epoch-e))*val
                cum_gamma[idx] += self.gamma**(current_epoch-e)
        
        for g in cum_gamma:
            if g == 0.0:
                g+=1e-6
        self.mu = self.mu/cum_gamma
        return self.mu
    
    def Train(self,lr = 1e-3,llr = 1e-3,max_epoches = 100,schedule_lr = False,schedule_t = None,schedule_gamma = 0.1,update_mean=False,verbose=True):
        """
        Train hyperparameters(Covariance,noise) of GPR
        data : In shape as [Group,index,value,noise]
        method : {'MML','LOO','NNP'}
            MML:maximize log marginal likelihood
            LOO:maximize Leave-One-Out cross-validation predictive probability 
        """
        
        matrix_params,sigma_params = self.Parameter_Groups()
        optimizer = torch.optim.Adam([{'params':matrix_params,'lr':lr},
                                    {'params':sigma_params,'lr':llr}], lr=lr,weight_decay=0.0)
        if schedule_lr:
            lr_scd = torch.optim.lr_scheduler.MultiStepLR(optimizer,schedule_t,gamma = schedule_gamma)

        if update_mean:
            self.mu = self.MLE_Mean()
            #print(self.mu)
        current_epoch = max(self.data.keys())
        for epoch in range(max_epoches):
            self.zero_grad()
            loss = 0.0
            for e in self.data.keys():
                if self.loss_type == 'LOO':
                    loss -= self.Log_LOO_Predictive_Probability(self.data[e])*(self.gamma**(current_epoch-e))
                elif self.loss_type == 'MML':
                    loss -= self.Log_Marginal_Likelihood(self.data[e])*(self.gamma**(current_epoch-e))
                else:
                    raise RuntimeError("Not supported training method!!")
            loss.backward()
            optimizer.step()
            if epoch%10==0 and verbose:
                print("Train_Epoch:{}\t|Sigma:{:.4f}\t|Loss:{:.4f}".format(epoch,torch.mean(torch.diagonal(self.Covariance())).detach().item(),loss.item()))

            if schedule_lr:
                lr_scd.step()
                
        return loss.item()
    
    def Predict_Loss(self,data,priori_idx,posteriori_idx):
        for p in priori_idx:
            if p in posteriori_idx:
                posteriori_idx.remove(p) # do not predict the measured idx
        mu_p,sigma_p = self.Posteriori(data[priori_idx,:])

        pdist = MultivariateNormal(loc = mu_p[posteriori_idx],covariance_matrix = sigma_p[posteriori_idx,:][:,posteriori_idx])
        predict_loss = -pdist.log_prob(torch.tensor(data[posteriori_idx,1],device=self.device,dtype=torch.float))
        predict_loss = predict_loss.detach().item()
        return predict_loss,mu_p,sigma_p
        
    def Select_Clients(self,number=10,epsilon = 0.0,weights = None,Dynamic=False,Dynamic_TH=0.0):
        """
        Select the clients which may lead to the maximal loss decrease
        Sequentially select the client and update the postieriori
        """
        def max_loss_decrease_client(client_group,Sigma,weights = None):
            Sigma_valid = Sigma[:,client_group][client_group,:]
            Diag_valid = self.discount[client_group]/torch.sqrt(torch.diagonal(Sigma_valid)) # alpha_k/sigma_k

            if weights is None:
                total_loss_decrease = torch.sum(Sigma_valid,dim=0)*Diag_valid
            else:
                # sum_i Sigma_ik*p_i
                total_loss_decrease = torch.sum(torch.tensor(weights[client_group],device=self.device,dtype=torch.float).reshape([len(client_group),1])*Sigma_valid,dim=0)*Diag_valid
            mld,idx = torch.max(total_loss_decrease,0)
            idx = idx.item()
            selected_idx = client_group[idx]
            p_Sigma = Sigma-Sigma[:,selected_idx:selected_idx+1].mm(Sigma[selected_idx:selected_idx+1,:])/(Sigma[selected_idx,selected_idx])

            return selected_idx,p_Sigma,mld.item()

        prob = np.random.rand(1)[0]
        if prob<epsilon:
            # use epsilon-greedy
            return None
        else:
            Sigma = self.Covariance()
            remain_clients = list(range(self.num_users))
            selected_clients = []
            for i in range(number):  
                idx,Sigma,total_loss_decrease = max_loss_decrease_client(remain_clients,Sigma,weights)
                if Dynamic and -total_loss_decrease<Dynamic_TH:
                    break
                selected_clients.append(idx)
                remain_clients.remove(idx)
            
            return selected_clients
    

    def Reset_Discount(self):
        self.discount = torch.ones(self.num_users,device=self.device).detach()

    def Update_Discount(self,index,factor=0.9):
        self.discount[index]*=factor



                


 

class SE_Kernel(torch.nn.Module):
    """
    Module to calculate the Squared Exponential Kernel.
    """
    def __init__(self,init_sigma = 1.0,l = 1.0,device=torch.device('cpu')):
        super(SE_Kernel, self).__init__()
        self.sigma_f = Parameter(torch.tensor(init_sigma))
        self.l = l
        self.device=device

    def forward(self,xs):
        """
        Return the Sigma(Covariance Matrix) with the given distance matrix r
        where r_{ij} = (x_i-x_j)'(x_i-x_j) 
        """
        x_size = xs.size(1)
        A = torch.sum(xs**2,dim=0,keepdim=True)*torch.ones([x_size,x_size],device=self.device)
        R = A+A.transpose(0,1)-2*(xs.transpose(0,1)).mm(xs)
        return torch.exp(-0.5*R/self.l)*self.sigma_f**2


class Poly_Kernel(torch.nn.Module):
    """
    Module to calculate the Polynomial Kernel
    """
    def __init__(self,order = 1,Normalize = False,device=torch.device('cpu')):
        super(Poly_Kernel,self).__init__()
        self.order = order
        self.Normalize = Normalize
        
        # Add an additional variance parameter
        self.sigma_f = Parameter(torch.tensor(1.0))

        self.device=device
        
    def forward(self,xs):
        """
        Return the covariance of x = [x1,x2]
        """
        k = (xs.transpose(0,1)).mm(xs)
        if self.Normalize:
            # Make K(x,x) = 1
            x_size = xs.size(1)
            A = torch.sum(xs**2,dim=0,keepdim=True)*torch.ones([x_size,x_size],device=self.device)
            k = k/torch.sqrt(A)/torch.sqrt(A.transpose(0,1))
            return torch.pow(k,self.order)*self.sigma_f**2
        else:
            return torch.pow(k,self.order)



class Kernel_GPR(GPR):
    """
    A GPR class with covariance defined by a kernel function

    Parameters:
        Projection.PMatrix: A Matrix that projects index (in a one-hot vector form)
                            into a low-dimension space. 
                            In fact each column of this matrix corresponds to the location 
                            of that user in the low-dimension space.
        Kernel.sigma_f: Diagonal of covariance matrix, which reveals the uncertainty 
                        priori on each user.We assume the same uncertainty before sampling.
         
        noise: Standard Deviation of sample noise (sigma_n). 
               This noise is caused by averaging weights, 
               and we assume the same noise for all clients.
        
        Total number of parameters is num_users x dimension + 2
    """

    
    def __init__(self,num_users,loss_type = 'LOO',init_noise = 0.01,reusable_history_length=10,gamma=1.0,device = torch.device('cpu'),dimension = 10,kernel = SE_Kernel,**Kernel_Arg):
        class Index_Projection(torch.nn.Module):
            """
            Module that project an index(an int between 0 and num_users-1) to a dimension-D space
            """
            def __init__(self, num_users,dimension=10):
                super(Index_Projection, self).__init__()
                # Normalize the initialization so that the mean of ||x|| is 1
                self.PMatrix = Parameter(torch.randn(dimension,num_users)/np.sqrt(dimension))
                # self.PMatrix = Parameter(torch.ones(dimension,num_users))
            def forward(self,i):
                """
                Return a column vector as the location in the dimension-D space
                """
                return self.PMatrix[:,i]
        super(Kernel_GPR, self).__init__(num_users,loss_type,init_noise,reusable_history_length,gamma,device)
        self.Projection = Index_Projection(num_users,dimension)
        self.Kernel = kernel(device=device,**Kernel_Arg)


    def Set_Parameters(self,mu=None,proj=None,sigma = None,noise = None):
        if mu is not None:
            self.mu = mu
        if proj is not None:
            self.Projection.PMatrix.data = proj
        if sigma is not None:
            self.Kernel.sigma_f.data = sigma
        if noise is not None:
            self.noise = noise
        

    def Covariance(self,ids = None):
        """
        Return the Covariance Matrix at the given indexes
        """
        if ids is None:
            # Calculate the covariance matrix of all users by default
            ids = list(range(self.num_users))
        xs = self.Projection(ids)
        return self.Kernel(xs)+(self.noise**2)*torch.eye(len(ids),device=self.device)
        

        
    def Parameter_Groups(self):
        proj_parameters = [self.Projection.PMatrix,]
        sigma_parameters = [self.Kernel.sigma_f,] if hasattr(self.Kernel,'sigma_f') else None
        return proj_parameters,sigma_parameters




class Matrix_GPR(GPR):
    """
    A GPR class with covariance defined by a positive definite matrix Sigma

    Parameters:
        Lower: Elements of the lower triangular matrix L except the diagonal
        
        Diagonal: |Diagonal| will be the diagonal elements of L 

        The Covariance Matrix Priori is computed as LL'.
        The total number of parameters is (num_users*num_users+num_users)//2+1
    """
    def __init__(self,num_users,loss_type = 'LOO',init_noise = 0.01,reusable_history_length=10,gamma=1.0,device = torch.device('cpu')):
        super(Matrix_GPR, self).__init__(num_users,loss_type,init_noise,reusable_history_length,gamma,device=device)
        # Lower Triangular Matrix L Elements without diagonal elements
        self.Lower = Parameter(torch.zeros((num_users*num_users-num_users)//2))
        self.index = torch.zeros((num_users*num_users-num_users)//2,dtype = torch.long)
        n = 0
        for i in range(num_users):
            for j in range(num_users):
                if j<i:# an lower triangular matrix 
                    self.index[n]=i*num_users+j
                    n+=1
                else:
                    break
        # Diagonal elements of L
        self.Diagonal = Parameter(torch.ones(num_users))

    def Set_Parameters(self,mu=None,diag=None,noise = None,lower = None):
        if mu is not None:
            self.mu.copy_(mu)
        if diag is not None:
            self.Diagonal.data = diag
        if noise is not None:
            self.noise = noise
        if lower is not None:
            self.Lower.data = lower


    def Covariance(self,ids = None):
        """
        Return the Covariance Matrix according to Lower Trangular Matrix
        """
        L = torch.zeros(self.num_users*self.num_users,device=self.device)
        L.scatter_(0,self.index,self.Lower)
        L = L.reshape([self.num_users,self.num_users])
        L = L+torch.abs(torch.diag(self.Diagonal))#Now we get L
        # Sigma = LL'
        Sigma = L.mm(L.transpose(0,1))
        if ids is None:
            # Return the covariance matrix of all users by default
            return Sigma+(self.noise**2)*torch.eye(self.num_users,device=self.device)
        else:
            return Sigma[ids,:][:,ids]+(self.noise**2)*torch.eye(len(ids),device=self.device)

    def Parameter_Groups(self):
        matrix_parameters = [self.Lower,self.Diagonal]
        sigma_parameters = None
        return matrix_parameters,sigma_parameters

        


# if __name__=='__main__':
#     num_users = 30
#     gpr = Kernel_GPR(num_users,dimension = 1,init_noise=0.01,kernel=SE_Kernel,l = 1.0)

#     pmatrix = np.zeros([1,num_users])
    
#     pmatrix[0,:] = np.arange(num_users,dtype = np.float)/5
#     gpr.set_parameters(proj=torch.tensor(pmatrix))
#     sel = gpr.Select_Clients(number=3,discount_method='time',verbose=True)
#     print(sel)
#     plt.show()

    



