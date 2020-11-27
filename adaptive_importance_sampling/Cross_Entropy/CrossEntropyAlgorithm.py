

"""
Cross Entropy Algorithm 
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
# verbose : verbosity parameter (boolean)
# aux_distribution : auxiliary distribution of which the parameters will be optimized to compute probability of failure
# bounds : bounds on parameters of auxiliary distribution (ot.Interval)
# initial_theta : initial value of parameters of auxiliary distribution (list of floats)
# active_parameters : list of booleans which indicate if the parameters of aux_distribution need to be optimized


import numpy as np
import copy
import openturns as ot


## Container of CE results
class CEResult(ot.SimulationResult):
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
    
    def setCESamples(self,samples):
        self.Samples = samples
        return None
        
    def getAuxiliaryDensity(self):
        return self.aux_distrib 
    
    def setAuxiliaryDensity(self,density):
        self.aux_distrib  = density
        return None
        
class CrossEntropyAlgorithm(object):
    def __init__(self,event,n_IS,rho_quantile,aux_distribution,active_parameters,bounds,initial_theta,verbose = False):
        
        self.n_IS = n_IS
        self.limit_state_function = event.getFunction() #limit state function
        self.S = event.getThreshold() #Failure threshold
        self.dim = event.getAntecedent().getDimension() #dimension of input space
        self.proba = 0.
        self.distrib = event.getAntecedent().getDistribution() #initial distribution
        self.aux_distrib = copy.deepcopy(aux_distribution) #copy of auxiliary distribution
        self.active_parameters = np.array(active_parameters) #active parameters
        self.operator = event.getOperator() # event operator
        if self.operator(1,2)==True: 
            self.rho_quantile = rho_quantile/100 #definition of rho quantile if exceedance probability
        else:
            self.rho_quantile = 1- rho_quantile/100 #definition of rho quantile in case g<0
        self.nb_eval = 0 #Current number of evaluations
        self.verbose = verbose # verbosity parameters
        self.samples = None # Current samples
        self.bounds = bounds #bounds of the active parameters
        self.initial_theta = initial_theta #initial values of the active parameters
        self.dim_theta = len(initial_theta) #dimension of active parameters
        self.result = CEResult()
        
		#Check of active parameters list validity
        if len(self.active_parameters )!=len(self.aux_distrib.getParameter()):
            raise ValueError('Wrong number of active parameters')
			
        #Check of validity of initial theta vector
        if np.sum(self.active_parameters)!=len(initial_theta):
            raise ValueError('Wrong correspondance between the number of active parameters and the given initial vector of parameters')

        
    #definition of function that updates the auxiliary distribution based on new theta values
    def update_dist(self,theta): 
        theta_ = np.array(self.aux_distrib.getParameter())

        theta_[self.active_parameters] = theta
        self.aux_distrib.setParameter(theta_)
        return None
        
    #definition of objective function for Cross entropy
    def obj_func(self,Sample,Resp_sample,theta,quantile_courant):

        self.update_dist(theta) # update of auxiliary distribution
        y= np.array([self.operator(Resp_sample[i,0],quantile_courant[0]) for i in range(Resp_sample.getSize())]) #find failure points
        indices_critic=np.where(y==True)[0].tolist()  # find failure samples indices
        Resp_sample_critic = Resp_sample.select(indices_critic)
        Sample_critic = Sample.select(indices_critic) # select failure samples

        pdf_init_critic = self.distrib.computePDF(Sample_critic) #evaluate initial PDF on failure samples
        pdf_aux_critic = self.aux_distrib.computePDF(Sample_critic)#evaluate auxiliary PDF on failure samples

        f = 1/self.n_IS * np.sum(np.array([pdf_init_critic])/np.array([pdf_aux_critic])*np.log(pdf_aux_critic)) #calculation of objective function
        return [f]


	#main function that computes the failure probability
    def run(self):
        
        bounds = self.bounds
        if self.operator(self.S,self.S+1) == True:
            quantile_courant =ot.Point([self.S+1])
        else: 
            quantile_courant =ot.Point([self.S-1])
			
        theta_0 =self.initial_theta
        num_iter = 0
		#main loop of adaptive importance sampling
        while self.operator(self.S,quantile_courant[0]):
        
            theta_courant = theta_0
            self.update_dist(theta_0)
            Sample= self.aux_distrib.getSample(self.n_IS) # drawing of samples using auxiliary density
            Resp_sample = self.limit_state_function(Sample) #evaluation on limit state function
            quantile_courant = Resp_sample.computeQuantile(self.rho_quantile) #computation of current quantile
            
            f_opt = lambda theta : self.obj_func(Sample,Resp_sample,theta,quantile_courant) #definition of objective function for CE

            objective = ot.PythonFunction(self.dim_theta, 1, f_opt)

            problem = ot.OptimizationProblem(objective) # Definition of CE optimization  problemof auxiliary distribution parameters
            problem.setBounds(bounds)
            problem.setMinimization(False)

            algo_optim = ot.Dlib(problem,'Global')
            algo_optim.setMaximumIterationNumber(50000) 
            algo_optim.setStartingPoint(theta_0)
            algo_optim.run() #Run of CE optimization
			
			
            # retrieve results
            result = algo_optim.getResult()
            theta_0 = result.getOptimalPoint()
            if self.verbose == True :
                if num_iter == 0:
                    print('theta', '| current threshold')
                    print(theta_0,'|',quantile_courant[0])
                else:
                    print(theta_0,'|',quantile_courant[0])
                    
            num_iter+=1
            
        #Estimate probability
        self.update_dist(theta_courant) #update of auxiliary density 
        y= np.array([self.operator(Resp_sample[i][0],self.S) for i in range(Resp_sample.getSize())]) #find failure points
        indices_critic=np.where(y==True)[0].tolist() # find failure samples indices
        
        Resp_sample_critic = Resp_sample.select(indices_critic) 
        Sample_critic = Sample.select(indices_critic)

        pdf_init_critic = self.distrib.computePDF(Sample_critic) #evaluate initial PDF on failure samples
        pdf_aux_critic = self.aux_distrib.computePDF(Sample_critic)#evaluate auxiliary PDF on failure samples
        proba = 1/self.n_IS * np.sum(np.array([pdf_init_critic])/np.array([pdf_aux_critic])) #Calculation of failure probability
        
        self.proba = proba
        self.samples = Sample
        
        # Save of data in SimulationResult structure
        self.result.setProbabilityEstimate(proba)
        self.result.setCESamples(Sample)
        self.result.setAuxiliaryDensity(self.aux_distrib)
        return None
    
    #Accessor to results
    def getResult(self):
        return self.result
    
        