'''
Copyright (c) 2019. IIP Lab, Wuhan University
'''

from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow_probability import distributions as tfp


class AddGaussianLoss(layers.Layer):
    '''
        Computed weighted KL divergence loss between 
        variational gaussian distribution and prior 
        standard normal distribution to serve as the
        information bottleneck.
    '''
    def __init__(self, 
                 **kwargs):
        super(AddGaussianLoss, self).__init__(**kwargs)                
        self.lamb_kl  = self.add_weight(shape=(), 
                                        name="lamb_kl", 
                                        trainable=False)

    def call(self, inputs):
        mu, std  = inputs
        var_dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        pri_dist = tfp.MultivariateNormalDiag(loc=K.zeros_like(mu), 
                                                      scale_diag=K.ones_like(std))    
        kl_loss  = self.lamb_kl*K.mean(tfp.kl_divergence(var_dist, pri_dist))
        return kl_loss


class ReparameterizeGaussian(layers.Layer):
    '''
        Reparameterization trick for Gaussian distribution
    '''
    def __init__(self, **kwargs):
        super(ReparameterizeGaussian, self).__init__(**kwargs)

    def call(self, stats):
        mu, std = stats
        dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        return dist.sample()


class ProductOfExpertGaussian(layers.Layer):
    '''
        Compute the Product-of-Experts Gaussian embedding
        from the modality-specifc hidden representations
    '''
    def __init__(self, **kwargs):
        super(ProductOfExpertGaussian, self).__init__(**kwargs)    

    def call(self, inputs):
        mu_list, std_list = zip(*inputs)
        prec_list = [(1/(std**2)) for std in std_list]
        
        ### mu is weighted by the precision of each expert
        poe_mu  = K.sum([mu*prec/K.sum(prec_list, axis=0) 
                        for mu, prec in zip(mu_list, prec_list)], axis=0)
        poe_std = K.sqrt(1/K.sum(prec_list, axis=0))
        return [poe_mu, poe_std]


if __name__ == "__main__":
    '''
        For test purpose ONLY
    '''
    pass