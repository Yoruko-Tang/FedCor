import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import pickle
import copy
import math

import matplotlib
import matplotlib.pyplot as plt


class GPR(torch.nn.Module):
    """
    Gaussian Process Regression Model For Federated Learning
    This is a base class and a Covariance() function must be implemented by its subclasses

    Parameters:     
        noise: Standard Deviation of sample noise (sigma_n). 
               This noise is caused by averaging weights, 
               and we assume the same noise for all clients.
    
    Non-Parameter Tensor:
        mu: Mean Priori, which is fixed while training. We set it as the postieriori of the 
            last round.
    """
    def __init__(self,num_users,loss_type = 'LOO',init_noise = 1.0):
        """
        Arguments:
            num_users: Number of users in a Federated Learning setting
            loss_type: {"LOO","MML","NNP"}
        """
        super(GPR, self).__init__()
        self.num_users = num_users
        self.loss_type = loss_type
        # sigma_n
        self.noise = Parameter(torch.tensor(init_noise))
        self.mu = torch.zeros(num_users).detach()
        self.loss_stat = torch.ones(num_users).detach()
        self.discount = torch.ones(num_users).detach()

    def Covariance(self,ids = None):
        raise NotImplementedError("A GPR class must have a function to calculate covariance matrix")

    def Posteriori(self,data):
        """
        Get the posteriori with the data
        data: given in the form [index,loss,noisy = {0,1}]
        return:mu|data,Sigma|data
        """
        data = torch.tensor(data).to(self.noise)
        indexes = data[:,0].long()
        values = data[:,1]
        noisy = data[:,2]
        Cov = self.Covariance()
        
        Sigma_inv = torch.inverse(Cov[indexes,:][:,indexes]+torch.diag(noisy).to(self.noise)*(self.noise**2))
        # e,v = torch.symeig(Sigma_inv)
        # print(e)
        # print((Cov[indexes,:][:,indexes]+torch.diag(noisy)*(self.noise**2)).mm(Sigma_inv))
        mu = self.mu.to(self.noise)+((Cov[:,indexes].mm(Sigma_inv)).mm((values-self.mu[indexes].to(self.noise)).unsqueeze(1))).squeeze()
        Sigma = Cov-(Cov[:,indexes].mm(Sigma_inv)).mm(Cov[indexes,:])
        return mu.detach(),Sigma.detach()

    def Log_Marginal_Likelihood(self,data):
        """
        MML:
        Calculate the log marginal likelihood of the given data
        data: given in the form [index,loss,noisy = {0,1}]
        return log(p(loss|mu,sigma,relation,sigma_n))
        """
        data = torch.tensor(data).to(self.noise)
        indexes = data[:,0].long()
        values = data[:,1]
        noisy = data[:,2]
        mu = self.mu[indexes].to(self.noise)
        Sigma = self.Covariance(indexes)+torch.diag(noisy).to(self.noise)*(self.noise**2)
        distribution = MultivariateNormal(loc = mu,covariance_matrix = Sigma)
        res = distribution.log_prob(values)

        return res

    def Log_LOO_Predictive_Probability(self,data):
        """
        LOO:
        Calculate the Log Leave-One-Out Predictive Probability of the given data
        data: given in the form [index,loss,noisy = {0,1}]
        return: \sum log(p(y_i|y_{-i},mu,sigma,relation,sigma_n))
        """

        # High efficient algorithm exploiting partitioning
        data = torch.tensor(data).to(self.noise)
        log_p = 0.0
        indexes = data[:,0].long()
        values = data[:,1]
        noisy = data[:,2]
        Sigma_inv = torch.inverse(self.Covariance(indexes)+torch.diag(noisy).to(self.noise)*(self.noise**2))
        K_inv_y = (Sigma_inv.mm((values-self.mu[indexes].to(self.noise)).unsqueeze(1))).squeeze()
        for i in range(len(data)):
            mu = values[i]-K_inv_y[i]/Sigma_inv[i,i]
            sigma = torch.sqrt(1/Sigma_inv[i,i])
            # print(mu,sigma)
            dist = Normal(loc = mu,scale = sigma)
            # if dist.log_prob(values[i])>0:
            #     print(mu,sigma,values[i],dist.log_prob(values[i]))
            #     input()
            log_p+=dist.log_prob(values[i])
        
        return log_p

    def Log_NonNoise_Predictive_Error(self,data):
        """
        NNP:
        When getting some noisy data and some non-noisy data,we are able to calculate the 
        log predictive error(l2 norm) of the non-noisy data(target) with the observation of the noisy 
        data(input). 
        """
        data = torch.tensor(data).to(self.noise)
        noisy_data = data[data[:,2]==1]
        nonnoisy_data = data[data[:,2]==0]
        if len(nonnoisy_data) == 0:
            print(data)
            raise RuntimeError("No non-noisy data!")
        noisy_indexes = noisy_data[:,0].long()
        noisy_values = noisy_data[:,1]
        nonnoisy_indexes = nonnoisy_data[:,0].long()
        nonnoisy_values = nonnoisy_data[:,1]
        wCovariance = self.Covariance()
        Sigma_inv = torch.inverse(wCovariance[noisy_indexes,:][:,noisy_indexes]+torch.eye(len(noisy_data)).to(self.noise)*(self.noise**2))
        Noisy_NonNoisy_Covariance = wCovariance[nonnoisy_indexes,:][:,noisy_indexes]
        post_mu = self.mu[nonnoisy_indexes].to(self.noise)+(Noisy_NonNoisy_Covariance.mm(Sigma_inv).mm((noisy_values-self.mu[noisy_indexes].to(self.noise)).unsqueeze(1))).squeeze()
        # post_covariance = wCovariance[nonnoisy_indexes,:][:,nonnoisy_indexes]-Noisy_NonNoisy_Covariance.mm(Sigma_inv).mm(Noisy_NonNoisy_Covariance.transpose(0,1))
        # dist = MultivariateNormal(loc = post_mu,covariance_matrix = post_covariance)
        mseloss = torch.nn.MSELoss()
        return mseloss(post_mu,nonnoisy_values)
        # return dist.log_prob(nonnoisy_values)

    def Parameter_Groups(self):
        raise NotImplementedError("A GPR class must have a function to get parameter groups = [Mpar,Spar]")
        
    def Select_Clients(self,number=10,loss_power = 0.5,epsilon = 0.0,discount_method = 'loss',weights = None,Dynamic=False,Dynamic_TH=0.0,verbose = False):
        """
        Select the clients which may lead to the maximal loss decrease
        Sequentially select the client and update the postieriori
        """
        def max_loss_decrease_client(client_group,Sigma,power = 0.3,discount_method = 'loss',weights = None):
            # print(mu)
            Sigma_valid = Sigma[:,client_group]
            Diag_valid = 1.0/(torch.diagonal(Sigma[:,client_group][client_group,:])+self.noise**2)
            # Diag_valid = Diag_valid*(self.loss_decrease_estimation[client_group].to(self.noise)-mu[client_group])
            Diag_valid = -Diag_valid*torch.sqrt(torch.diagonal(Sigma[:,client_group][client_group,:]))
            if discount_method=='loss':
                Diag_valid = Diag_valid*torch.pow(self.loss_stat[client_group],power)
            elif discount_method=='time':
                Diag_valid = Diag_valid*self.discount[client_group]

            # loss_decrease = Sigma_valid*Diag_valid
            if weights is None:
                total_loss_decrease = torch.sum(Sigma_valid,dim=0)*Diag_valid
            else:
                total_loss_decrease = torch.sum(torch.tensor(weights).reshape([self.num_users,1])*Sigma_valid,dim=0)*Diag_valid
            # total_loss_decrease = torch.sum(loss_decrease,dim = 0)# Add across row
            mld,idx = torch.min(total_loss_decrease,0)
            idx = idx.item()
            selected_idx = client_group[idx]
            p_Sigma = Sigma-Sigma[:,selected_idx:selected_idx+1].mm(Sigma[selected_idx:selected_idx+1,:])/(Sigma[selected_idx,selected_idx]+self.noise**2)
            d_mu = Sigma_valid[:,idx]*Diag_valid[idx]
            # return selected_idx,mu+loss_decrease[:,idx],p_Sigma
            return selected_idx,p_Sigma,mld.item(),d_mu.detach()


        # mu = self.mu
        prob = np.random.rand(1)[0]
        if prob<epsilon:
            # use epsilon-greedy
            return None
        else:
            Sigma = self.Covariance()
            if verbose:
                y_major_locator = plt.MultipleLocator(0.5)
                plt.figure(figsize=(8,8))
                mu = self.mu
                std = torch.sqrt(torch.diagonal(Sigma)).detach()
                plt.plot(range(len(mu)),mu.numpy(),'b-')
                plt.fill_between(range(len(mu)),mu.numpy()-std.numpy(),mu.numpy()+std.numpy(),alpha=0.3)
                plt.ylabel('Predictive Loss Change',fontsize=26)
                plt.xlabel('Client Index',fontsize=26)
                plt.tick_params(labelsize=20)
                plt.gca().yaxis.set_major_locator(y_major_locator)
            remain_clients = list(range(self.num_users))
            selected_clients = []
            for i in range(number):
                if verbose:
                    comp_idx,_,comp_tld,comp_ld = max_loss_decrease_client(np.random.choice(remain_clients,1),Sigma,loss_power,discount_method,weights)
                    plt.arrow(comp_idx,mu[comp_idx].item(),0,-std[comp_idx].item(),length_includes_head=True,width = 0.1,head_width = 0.3,head_length = 0.1,ec = 'k',fc = 'k')
                    plt.plot(comp_idx,mu[comp_idx].item()-std[comp_idx].item(),'kx',markersize = 26)
                    comp_mu = mu+comp_ld
                    plt.plot(range(len(comp_mu)),comp_mu.numpy(),'k-',label='Selection:{}, Mean:{:.3f}'.format(comp_idx,np.mean(comp_mu.numpy())))
                    #plt.plot(range(len(mu)),np.ones(len(mu))*np.mean(mu.numpy()),'b:')
                    #plt.plot(range(len(comp_mu)),np.ones(len(comp_mu))*np.mean(comp_mu.numpy()),'k:')
                    print('compared loss decrease:',comp_tld)
                idx,Sigma,total_loss_decrease,loss_decrease = max_loss_decrease_client(remain_clients,Sigma,loss_power,discount_method,weights)
                if verbose:
                    plt.arrow(idx,mu[idx].item(),0,-std[idx].item(),length_includes_head=True,width = 0.1,head_width = 0.3,head_length = 0.1,ec = 'r',fc = 'r')
                    plt.plot(idx,mu[idx].item()-std[idx].item(),'rx',markersize = 26)
                    std = torch.sqrt(torch.diagonal(Sigma)).detach()
                    mu = mu+loss_decrease
                    plt.plot(range(len(mu)),mu.numpy(),'r-',label='Selection:{}, Mean:{:.3f}'.format(idx,np.mean(mu.numpy())))
                    plt.legend(fontsize=26,loc='upper center')
                    #plt.plot(range(len(mu)),np.ones(len(mu))*np.mean(mu.numpy()),'r:')
                    plt.tight_layout()
                    # plt.legend(loc = 'lower center',ncol=3, borderaxespad=0.,fontsize = 20)
                    # plt.savefig(save_path+'/test_accuracy.png')
                    plt.figure(figsize=(8,8))
                    plt.plot(range(len(mu)),mu.numpy(),'b-')
                    plt.fill_between(range(len(mu)),mu.numpy()-std.numpy(),mu.numpy()+std.numpy(),alpha=0.3)
                    for u in selected_clients:
                        plt.plot(u,mu[u].item(),'rx',markersize = 26)
                    plt.plot(idx,mu[idx].item(),'rx',markersize = 26)
                    plt.ylabel('Predictive Loss Change',fontsize=26)
                    plt.xlabel('Client Index',fontsize=26)
                    plt.gca().yaxis.set_major_locator(y_major_locator)
                    plt.tick_params(labelsize=20)
                    print('loss decrease:',total_loss_decrease)
                if Dynamic and -total_loss_decrease<Dynamic_TH:
                    break
                selected_clients.append(idx)
                remain_clients.remove(idx)
            if verbose:
                plt.tight_layout()

            return selected_clients

    def update_loss(self,data):
        data = torch.tensor(data).to(self.noise)
        indexes = data[:,0].long()
        values = data[:,1]
        self.loss_stat[indexes]=values
        # self.loss_decrease_estimation[indexes]=self.loss_decrease_estimation[indexes]*0.3+values*0.7
        # print(self.loss_decrease_estimation[indexes])

    # def update_dloss(self,idxs,discount_factor = 0.95):
    #     self.loss_stat[idxs]*=discount_factor

    def Predict_Loss(self,data,priori_idx,posteriori_idx):
        mu_p,sigma_p = self.Posteriori(data[priori_idx,:])
        noise_scale = 1e-3
        while True:
            try:
                pdist = MultivariateNormal(loc = mu_p[posteriori_idx],
                                           covariance_matrix = sigma_p[posteriori_idx,:][:,posteriori_idx]+noise_scale*torch.eye(len(posteriori_idx)))
                break
            except RuntimeError:
                noise_scale*=10
        predict_loss = -pdist.log_prob(torch.tensor(data[posteriori_idx,1]).to(mu_p))
        predict_loss = predict_loss.detach().item()
        return predict_loss,mu_p,sigma_p

    def Reset_Discount(self):
        self.discount = torch.ones(self.num_users).detach()

    def update_discount(self,index,factor=0.9):
        self.discount[index]*=factor



                


def TrainGPR(gpr,data,method = None,lr = 1e-3,llr = 1e-4,gamma = 0.9,max_epoches = 100,schedule_lr = False,schedule_t = None,schedule_gamma = 0.1,verbose=True):
    """
    Train hyperparameters(Covariance,noise) of GPR
    data : In shape as [Group,index,value,noise]
    method : {'MML','LOO','NNP'}
        MML:maximize log marginal likelihood
        LOO:maximize Leave-One-Out cross-validation predictive probability 
    """
    if method is not None:
        gpr.loss_type = method
    matrix_params,sigma_params = gpr.Parameter_Groups()
    optimizer = torch.optim.Adam([{'params':matrix_params,'lr':lr},
                                  {'params':sigma_params,'lr':llr}], lr=lr,weight_decay=0.0)
    if schedule_lr:
        lr_scd = torch.optim.lr_scheduler.MultiStepLR(optimizer,schedule_t,gamma = schedule_gamma)
    # for p in self.parameters():
    #     print(p)
    old_loss = None
    for epoch in range(max_epoches):
        gpr.zero_grad()
        loss = 0.0
        for group in range(len(data)):
            if gpr.loss_type == 'LOO':
                loss = loss*gamma - gpr.Log_LOO_Predictive_Probability(data[group])
            elif gpr.loss_type == 'MML':
                loss = loss*gamma - gpr.Log_Marginal_Likelihood(data[group])
            elif gpr.loss_type == 'NNP':
                loss = loss*gamma + gpr.Log_NonNoise_Predictive_Error(data[group])
            else:
                raise RuntimeError("Not supported training method!!")
        # if old_loss is not None:
        #     if loss.item()>old_loss:
        #         break
        loss.backward()
        optimizer.step()
        if epoch%10==0 and verbose:
            print("Train_Epoch:{}\t|Noise:{:.4f}\t|Sigma:{:.4f}\t|Loss:{:.4f}".format(epoch,gpr.noise.detach().item(),torch.mean(torch.diagonal(gpr.Covariance())).detach().item(),loss.item()))
            #print(loss)
        old_loss = loss.item()
        if schedule_lr:
            lr_scd.step()
            
    return loss.item() 

class SE_Kernel(torch.nn.Module):
    """
    Module to calculate the Squared Exponential Kernel.
    """
    def __init__(self,init_sigma = 1.0,l = 1.0):
        super(SE_Kernel, self).__init__()
        self.sigma_f = Parameter(torch.tensor(init_sigma))
        self.l = l

    def forward(self,xs):
        """
        Return the Sigma(Covariance Matrix) with the given distance matrix r
        where r_{ij} = (x_i-x_j)'(x_i-x_j) 
        """
        x_size = xs.size(1)
        A = torch.sum(xs**2,dim=0,keepdim=True)*torch.ones(x_size,x_size).to(self.sigma_f)
        R = A+A.transpose(0,1)-2*(xs.transpose(0,1)).mm(xs)
        return torch.exp(-0.5*R/self.l)*self.sigma_f**2


class Poly_Kernel(torch.nn.Module):
    """
    Module to calculate the Polynomial Kernel
    """
    def __init__(self,order = 1,Normalize = False):
        super(Poly_Kernel,self).__init__()
        self.order = order
        self.Normalize = Normalize
        
        # Add an additional variance parameter
        self.sigma_f = Parameter(torch.tensor(1.0))

        # print(self.order,self.Normalize)
        
    def forward(self,xs):
        """
        Return the covariance of x = [x1,x2]
        """
        k = (xs.transpose(0,1)).mm(xs)
        if self.Normalize:
            # Make K(x,x) = 1
            x_size = xs.size(1)
            A = torch.sum(xs**2,dim=0,keepdim=True)*torch.ones(x_size,x_size).to(self.sigma_f)
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

    
    def __init__(self, num_users,init_noise = 1.0,dimension = 10,kernel = SE_Kernel,loss_type = 'LOO',**Kernel_Arg):
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
        super(Kernel_GPR, self).__init__(num_users,loss_type,init_noise)
        self.Projection = Index_Projection(num_users,dimension)
        self.Kernel = kernel(**Kernel_Arg)


    def set_parameters(self,mu=None,proj=None,sigma = None,noise = None):
        if mu is not None:
            self.mu = mu
        if proj is not None:
            self.Projection.PMatrix.data = proj
        if sigma is not None:
            self.Kernel.sigma_f.data = sigma
        if noise is not None:
            self.noise.data = noise
        

    def Covariance(self,ids = None):
        """
        Return the Covariance Matrix at the given indexes
        """
        if ids is None:
            # Calculate the covariance matrix of all users by default
            ids = list(range(self.num_users))
        xs = self.Projection(ids)
        return self.Kernel(xs)
        

        
    def Parameter_Groups(self):
        proj_parameters = [self.Projection.PMatrix,]
        sigma_parameters = [self.Kernel.sigma_f,self.noise] if hasattr(self.Kernel,'sigma_f') else [self.noise,]
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
    def __init__(self, num_users,loss_type = 'LOO'):
        super(Matrix_GPR, self).__init__(num_users,loss_type)
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

    def set_parameters(self,mu=None,diag=None,noise = None,lower = None):
        if mu is not None:
            self.mu.copy_(mu)
        if diag is not None:
            self.Diagonal.data = diag
        if noise is not None:
            self.noise.data = noise
        if lower is not None:
            self.Lower.data = lower


    def Covariance(self,ids = None):
        """
        Return the Covariance Matrix according to Lower Trangular Matrix
        """
        L = torch.zeros(self.num_users*self.num_users).to(self.Lower)
        L.scatter_(0,self.index,self.Lower)
        L = L.reshape([self.num_users,self.num_users])
        L = L+torch.abs(torch.diag(self.Diagonal))#Now we get L
        # Sigma = LL'
        Sigma = L.mm(L.transpose(0,1))
        if ids is None:
            # Return the covariance matrix of all users by default
            return Sigma
        else:
            return Sigma[ids,:][:,ids]

    def Parameter_Groups(self):
        matrix_parameters = [self.Lower,self.Diagonal]
        sigma_parameters = [self.noise,]
        return matrix_parameters,sigma_parameters

        


if __name__=='__main__':
    num_users = 30
    gpr = Kernel_GPR(num_users,dimension = 1,init_noise=0.01,kernel=SE_Kernel,l = 1.0)

    pmatrix = np.zeros([1,num_users])
    
    pmatrix[0,:] = np.arange(num_users,dtype = np.float)/5
    gpr.set_parameters(proj=torch.tensor(pmatrix))
    sel = gpr.Select_Clients(number=3,discount_method='time',verbose=True)
    print(sel)
    plt.show()

    



