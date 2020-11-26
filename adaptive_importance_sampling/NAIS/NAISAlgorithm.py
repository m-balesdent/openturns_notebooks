
"""
Non parametric Adaptive Importance Sampling algorithm
Python implementation of adaptive Importance Sampling by Cross Entropy 
M. Balesdent and L. Brevault of ONERA, the French Aerospace Lab for the 
openTURNS consortium

source :  J. Morio & M. Balesdent, Estimation of Rare Event Probabilities 
in Complex Aerospace and Other Systems, A Practical Approach, Elsevier, 2015


"""



### Parameters of the class
# event : ThresholdEvent based on composite vector of input variables on limit state function 
# n_IS : number of IS samples at each step (integer)
# rho_quantile : percentage of points that are in the local failure domain (float between 0 and 100)


import numpy as np
import openturns as ot
import math as m


class NAISAlgorithm(object):
    def __init__(self,event,n_IS,rho_quantile):
        self.n_IS = n_IS
        self.limit_state_function = event.getFunction() #limit state function
        self.S = event.getThreshold() #Failure threshold
        self.dim = event.getAntecedent().getDimension() #dimension of input space
        self.proba = 0.
        self.distrib = event.getAntecedent().getDistribution() #initial distribution
        failure_condition =  event.getOperator() 
        if failure_condition(0,1) == True:
            self.rho_quantile = rho_quantile/100 #definition of rho quantile if exceedance probability
        else:
            self.rho_quantile = 1- rho_quantile/100 #definition of rho quantile in case g<0
        self.nb_eval = 0 #Current number of evaluations
        self.samples = None # Current input samples
        self.outputsamples = None  # Current output samples
        self.operator = failure_condition # event operator
        self.weights = None
        
	#function computing the auxiliary distribution as a function of current samples and associated weights
    def compute_aux_distribution(self,sample,weights):
    
        neff = np.sum(weights)**2 / np.sum(weights**2) #computation of weight
		
		# computation of bandwidth using Silverman rule
        silverman = sample.computeStandardDeviationPerComponent()*(neff * (self.dim + 2) / 4.)**(-1. / (self.dim + 4))
        
        margins = []
		
		# computation of auxiliary distribution using ot.Mixture 
        for k in range(self.dim):
            dist_coll= []
            for i in range(self.n_IS):
                dist_coll.append(ot.Normal(sample[i][k],silverman[k]))
            
            distri_margin = ot.Mixture(dist_coll,weights.tolist()[0])
            margins.append(distri_margin)      
        
        aux_distrib = ot.ComposedDistribution(margins)  
        return aux_distrib
    
	
	#function defining if samples is in failure mode
    def is_failure(self,Y,S_loc):
        
        if self.operator(0,1) == True:
            return np.array(Y)<S_loc
        else:
            return np.array(Y)>S_loc
          

	#function computing weigths  of sample
    def compute_weights(self,samples,resp_samples,S_loc,aux_distrib):
        
        f_value = self.distrib.computePDF(samples)
        g_value=aux_distrib.computePDF(samples)
        fraction = np.array(f_value)/np.array(g_value)
        weights = self.is_failure(resp_samples,S_loc).T*fraction
        
        return weights
                
	#main function that computes the failure probability    
    def compute_proba(self):
        
        k = 1
        sample = self.distrib.getSample(self.n_IS) # drawing of samples using initial density
        resp_sample = self.limit_state_function(sample) #evaluation on limit state function
        quantile_courant = resp_sample.computeQuantile(self.rho_quantile)[0] #computation of current quantile
        
        weights = self.compute_weights(sample,resp_sample,quantile_courant,self.distrib) #computation of weights
        aux_distrib = self.compute_aux_distribution(sample,weights) #computation of auxiliary distribution
        
		
		
        while self.operator(self.S,quantile_courant):
        
            sample = aux_distrib.getSample(self.n_IS) # drawing of samples using auxiliary density
            resp_sample = self.limit_state_function(sample) #evaluation on limit state function
            quantile_courant = resp_sample.computeQuantile(self.rho_quantile)[0] #computation of current quantile
            
            if self.operator(quantile_courant,self.S):
                quantile_courant = self.S
            else:
                weights = self.compute_weights(sample,resp_sample,quantile_courant,aux_distrib) #computation of weights
                aux_distrib = self.compute_aux_distribution(sample,weights) #update of auxiliary distribution

				
        #Estimation of failure probability
        y= np.array([self.operator(resp_sample[i][0],self.S) for i in range(resp_sample.getSize())])  #find failure points
        indices_critic=np.where(y==True)[0].tolist() # find failure samples indices
        
        resp_sample_critic = resp_sample.select(indices_critic)
        sample_critic = sample.select(indices_critic)

        pdf_init_critic = self.distrib.computePDF(sample_critic) #evaluate initial PDF on failure samples
        pdf_aux_critic = aux_distrib.computePDF(sample_critic) #evaluate auxiliary PDF on failure samples

        proba = 1/self.n_IS * np.sum(np.array([pdf_init_critic])/np.array([pdf_aux_critic])) #Calculation of failure probability
        self.proba = proba 
        self.samples = sample
        self.aux_distrib = aux_distrib
        return 
    
    
        #Accessor to the failure probability
    def getFailureProbability(self):
            return self.proba

        #Accessor to the IS samples
    def getCESamples(self):
            return self.samples
        
        #Accessor to the auxiliary distribution
    def getaux_density(self):
            return self.aux_distrib