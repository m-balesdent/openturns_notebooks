
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

## Container of NAIS results
class NAISResult(ot.SimulationResult):
    def __init__(self):
        self.ProbabilityEstimate = None
        self.Samples = None
        self.aux_distrib = None
        
    def getProbabilityEstimate(self):
        return self.ProbabilityEstimate
    
    def setProbabilityEstimate(self,proba):
        self.ProbabilityEstimate = proba
        return None
    
    def getSamples(self):
        return self.Samples
    
    def setNAISSamples(self,samples):
        self.Samples = samples
        return None
        
    def getAuxiliaryDensity(self):
        return self.aux_distrib 
    
    def setAuxiliaryDensity(self,density):
        self.aux_distrib  = density
        return None

class NAISAlgorithm(object):
    def __init__(self,event,n_IS,rho_quantile):
        self.n_IS = n_IS ##type : integer
        self.limit_state_function = event.getFunction() #limit state function ##type : Function
        self.S = event.getThreshold() #Failure threshold ##type : float
        self.dim = event.getAntecedent().getDimension() #dimension of input space ##type : integer
        self.proba = 0. ## type : float
        self.distrib = event.getAntecedent().getDistribution() #initial distribution ##type : ComposedDistribution
        range_ = self.distrib.getRange() #verification of unbounded distribution ## type : Interval
        if np.max(range_.getFiniteUpperBound())>0 or np.max(range_.getFiniteLowerBound())>0 : ## modif 23/07
            raise ValueError('Current version of NAIS is only adapted to unbounded distribution')
			
        failure_condition =  event.getOperator() ## type: ComparisonOperator
        if failure_condition(0,1) == True:
            self.rho_quantile = rho_quantile/100 #definition of rho quantile if exceedance probability ## type : float
        else:
            self.rho_quantile = 1- rho_quantile/100 #definition of rho quantile in case g<0
        self.nb_eval = 0 #Current number of evaluations ## Type : integer
        self.samples = None # Current input samples ## Type : Sample
        self.outputsamples = None  # Current output samples
        self.operator = failure_condition # event operator ## type: ComparisonOperator
        self.weights = None 
        self.result = NAISResult() ## type : NAISresult

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
            
            distri_margin = ot.Mixture(dist_coll,weights.tolist()) ## modif 23/07
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
        weights = self.is_failure(resp_samples,S_loc).T[0]*fraction.T[0] ## modif 23/07
        
        return weights
                
	#main function that computes the failure probability    
    def run(self):
        
        sample = self.distrib.getSample(self.n_IS) # drawing of samples using initial density ## type: Sample
        resp_sample = self.limit_state_function(sample) #evaluation on limit state function ## type : Sample
        quantile_courant = resp_sample.computeQuantile(self.rho_quantile)[0] #computation of current quantile ##type : float
        if self.operator(quantile_courant,self.S):
                quantile_courant = self.S
                weights = None
                aux_distrib = self.distrib
        else:
            weights = self.compute_weights(sample,resp_sample,quantile_courant,self.distrib) #computation of weights ## type : array
            aux_distrib = self.compute_aux_distribution(sample,weights) #computation of auxiliary distribution ##type : ComposedDistribution
        while self.operator(self.S,quantile_courant) and quantile_courant != self.S:
            sample = aux_distrib.getSample(self.n_IS) # drawing of samples using auxiliary density
            resp_sample = self.limit_state_function(sample) #evaluation on limit state function
            quantile_courant = resp_sample.computeQuantile(self.rho_quantile)[0] #computation of current quantile
            
            if self.operator(quantile_courant,self.S):
                quantile_courant = self.S
            else:
                weights = self.compute_weights(sample,resp_sample,quantile_courant,aux_distrib) #computation of weights
                aux_distrib = self.compute_aux_distribution(sample,weights) #update of auxiliary distribution

        #Estimation of failure probability
        y= np.array([self.operator(resp_sample[i][0],self.S) for i in range(resp_sample.getSize())])  #find failure points # type : array of boolean
        indices_critic=np.where(y==True)[0].tolist() # find failure samples indices
        
        resp_sample_critic = resp_sample.select(indices_critic) #type : Sample
        sample_critic = sample.select(indices_critic) # #type : Sample

        pdf_init_critic = self.distrib.computePDF(sample_critic) #evaluate initial PDF on failure samples # #type : Sample
        pdf_aux_critic = aux_distrib.computePDF(sample_critic) #evaluate auxiliary PDF on failure samples #type : Sample
        
        proba = 1/self.n_IS * np.sum(np.array([pdf_init_critic])/np.array([pdf_aux_critic])) #Calculation of failure probability #type : float
        self.proba = proba
        self.samples = sample
        self.aux_distrib = aux_distrib
        
        
        # Save of data in SimulationResult structure
        self.result.setProbabilityEstimate(proba)
        self.result.setNAISSamples(sample)
        self.result.setAuxiliaryDensity(self.aux_distrib)
        self.weights = weights ## modif 23/07
        self.outputsamples = resp_sample  ## modif 23/07        
        
        return None
    
    
    #Accessor to results
    def getResult(self):
        return self.result
