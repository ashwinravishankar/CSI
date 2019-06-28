from abc import ABC, abstractmethod

class ML_Model(ABC):
    """
    An abstract base class for ML model - Create, Train and Predict.  
    *args is a tuple of positional arguments, because the parameter name has * prepended.
    **kwargs is a dictionary of keyword arguments,because the parameter name has ** prepended.

    TODO: 
        - Add more helper methods to this class in the future
        - How to validate the model when training?
        - Check Estimator base class of TF to see what else need to be added in ABC

    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """ This the default constructor
        """
        raise NotImplementedError()

    @abstractmethod
    def set_params(self, *args, **kwargs):
        """ is called with supplied parameters as part of CREATE ML_MODEL
        """
        raise NotImplementedError()

    @abstractmethod
    def set_stats(self, *args, **kwargs):
        """ is called to supply the statistics needed for training the model
        """
        raise NotImplementedError()

    @abstractmethod
    def fit_train(self, *args, **kwargs):
        """ is called repeatedly for each tuple (with raw data, data scientist is required to normalize the features 
            if needed for that specific learning algorithm)
        """
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        """ is called repeated for all predict data tuples. For prediction purposes, 
            data scientist shall use the list of models supplied earlier by the ensemble() call. 
            Here the modeler is free to use whatever the ‘bagging’ method is appropriate (average, voting, etc).
        """
        raise NotImplementedError()

    @abstractmethod
    def serialize(self, *args, **kwargs):
        """ may be called at the end of training period.
        """
        raise NotImplementedError()

    @abstractmethod
    def deserialize(self, *args, **kwargs):
        """ may be called prior to prediction phase.
        """
        raise NotImplementedError()

    @abstractmethod
    def ensemble(self, *args, **kwargs):
        """ may be called to supply the models trained by other compute nodes. 
            This list of models shall be used to “predict” the target value.
        """
        raise NotImplementedError()


class ML_ModelException(Exception):
    """ Exception type used to raise exceptions within MLModel derived classes """
    def __init__(self,*args,**kwargs):
        Exception.__init__(self, *args, **kwargs)
    
