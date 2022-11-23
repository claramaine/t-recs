from operator import index

from torch import subtract
from trecs.metrics import Measurement
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import dirichlet
from scipy.special import rel_entr


# thesis:new (the whole thing)

def non_zero_array(prefs):
    small_prefs_mask = prefs < np.finfo(np.float64).tiny
    if(np.any(small_prefs_mask)):    
        np.place(prefs, small_prefs_mask, np.nextafter(np.float32(0), np.float32(1))) #replaces all zeros with a very tiny number just for the purposes of the entropy calculation
        prefs = prefs/np.sum(prefs) # normalize    
    return prefs
    

def kl_divergence(target, actual):
    if np.sum(rel_entr(target, actual)) == float('inf'):
       return np.sum(rel_entr(non_zero_array(target), non_zero_array(actual)))
    return np.sum(rel_entr(target, actual))

def mean_kld(targets, actuals):
    klds = []
    for t, a in zip(targets, actuals):
        kld = kl_divergence(t, a)
        if kld != float('inf'):
            klds.append(kld)
        else:
            print("oh no! kld for", targets, actuals, "could not be computed")
    return np.array(klds).mean()

class PreferenceChangeL2Norm(Measurement):
    def __init__(self, name="pref_change_l2", verbose=False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        start_prefs = recommender.users.actual_user_profiles.state_history[0]
        current_prefs = recommender.actual_user_profiles
        
        if start_prefs.shape != current_prefs.shape:
            self.observe(np.NaN)
            return
        
        # calculate the L2 norm of the difference between the starting and current preferences
        norm_dist = np.linalg.norm(np.subtract(start_prefs, current_prefs))
        self.observe(norm_dist)
        
class PreferenceChangeTVD(Measurement):
    def __init__(self, name="pref_change_tvd", verbose=False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        start_prefs = recommender.users.actual_user_profiles.state_history[0]
        current_prefs = recommender.actual_user_profiles
        
        if start_prefs.shape != current_prefs.shape:
            self.observe(np.NaN)
            return
        
        tvd = np.sum(np.abs(np.subtract(start_prefs, current_prefs))) / recommender.users_hat.num_attrs
        self.observe(tvd)
            
class PredictedTargetSimilarity(Measurement):
    def __init__(self, target_prefs, name="predicted_target_similarity", verbose=False):
        self.target_prefs = target_prefs
        #TODO: might possibly have to incorporate a timestep thing like MeanDistanceSimUsers in chaney_utils so we can account for a shifting target?
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        """
        The purpose of this metric is to calculate the average cosine similarity
        between the predictd peferences for a user and their target preferences.

        Sim(U, U_target) 

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """

        # user profiles in shape |U| x num_attributes
        predicted_user = recommender.predicted_user_profiles
        # target user profiles in shape |U| x num_attributes
        target_user = self.target_prefs #want an array for all the targets of all users 
        
        # this is usually the case for recsys which are not mf or content sim.
        if target_user.shape != predicted_user.shape:
            self.observe(np.NaN)
            return
        
        # calculate the cosine similarity between the predicted user preferences and the target user preferences
        sim_vals = cosine_similarity(predicted_user, target_user).diagonal()

        # to complete the measurement, call `self.observe(metric_value)`
        metric_value = sim_vals.mean()
        if metric_value == 0:
            self.observe(np.NaN) #so it's not graphed.
        else:
            self.observe(metric_value)

class ActualTargetSimilarity(Measurement):
    def __init__(self, target_prefs, name="actual_target_similarity", verbose=False):
        self.target_prefs = target_prefs
        #TODO: might possibly have to incorporate a timestep thing like MeanDistanceSimUsers in chaney_utils so we can account for a shifting target
        # since it would shift away from this thing we just passed. so we'd pass the whole big array and then have a timestep counter.
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        """
        The purpose of this metric is to calculate the average cosine similarity
        between the actual user peferences for a user and their target preferences.

        Sim(U^, U_target) 

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """

        # user profiles in shape |U| x num_attributes
        actual_user = recommender.actual_user_profiles
        # target user profiles in shape |U| x num_attributes
        target_user = self.target_prefs #want an array for all the targets of all users 
        
        if target_user.shape != actual_user.shape:
            self.observe(np.NaN) #cant deal with that rn need to find a different way to observe how J might affect sf
            return
        
        # calculate the cosine similarity between the predicted user preferences and the target user preferences
        sim_vals = cosine_similarity(actual_user, target_user).diagonal()

        # to complete the measurement, call `self.observe(metric_value)`
        metric_value = sim_vals.mean()
        if metric_value == 0:
            self.observe(np.NaN) #so it's not graphed.
        else:
            self.observe(metric_value)

class MostPrefferedAttributeChosen(Measurement):
    """
    Measures the proportion of times a user chose an item with (only) their most preffered attribute.

    Parameters
    -----------
        k: int
            The rank at which recall should be evaluated.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"most_preffered_attribute_chosen"``
            Name of the measurement component.
    """

    # Note: RecallMeasurement evalutes recall for the top-k (i.e., highest predicted value)
    # items regardless of whether these items derive from the recommender or from randomly
    # interleaved items. Currently, this metric will only be correct for
    # cases in which users iteract with one item per timestep

    def __init__(self, name="most_preffered_attribute_chosen", verbose=False):

        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        """
        Measures the proportion of relevant items (i.e., those users interacted with) falling
        within the top k ranked items shown..

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        
        def index_of_top_attr(attributes):
            return np.argmax(attributes, axis=1)

        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None)  # no interactions yet
            return

        else:
            #TODO
            # look at the interactions for each users
            # check, for the item from each interaction, if that item's attribute(s) 
            # is the same attribute as the user's most preffered attribute
            
            user_top_attrs = index_of_top_attr(recommender.actual_user_profiles) #do i want actual or do i want predicted??
            
            attributes_of_interactions = np.take(recommender.actual_item_attributes, recommender.interactions, axis=1).T
            
            b = np.ma.masked_inside(attributes_of_interactions, 0.1, 0.9)
            # if np.ma.count_masked(b) != 0: # this ensures that each item only has 1 attribute
            #     print("bastard attributes:", np.max(b[b.mask].data))
            
            item_top_attrs = index_of_top_attr(attributes_of_interactions)
            
            assert len(user_top_attrs) == len(item_top_attrs)
            
            #now compare how many of user_ta and item_ta are the same.
            percentage_same = np.sum(user_top_attrs != item_top_attrs)/len(user_top_attrs)
        
        self.observe(percentage_same)

class ActualUserProfiles(Measurement):
    """
    Records the value of user profiles.
    """
    def __init__(self, name="actual_user_profiles", verbose=False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender, **kwargs):
        user_profiles = recommender.actual_user_profiles.copy()
        self.observe(user_profiles)
        
class TargetUserProfiles(Measurement):
    """
    Records the value of user profiles.
    """
    def __init__(self, name="target_user_profiles", verbose=False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender, **kwargs):
        target_profiles = recommender.users.target_user_profiles.copy()
        self.observe(target_profiles)
    
# thesis:TODO these use dirichlet entropy which we don't want. results are computed manually in homo_plots notebooks but still would be nice to have as a
def avg_entropy(prefs_og):
    """ Takes creators array (row = creator, col = attributes)
        and calculates average Dirichlet entropy across all creators
    """
    prefs = prefs_og.copy().astype(np.longdouble)
    num = prefs.shape[0]
    
    small_prefs_mask = prefs < np.finfo(np.float64).tiny
    while(np.any(small_prefs_mask)):    
        np.place(prefs, small_prefs_mask, np.nextafter(np.float32(0), np.float32(1))) #replaces all zeros with a very tiny number just for the purposes of the entropy calculation
        sum1 = np.sum(prefs)
        prefs = prefs/np.sum(prefs)
        sum2 = np.sum(prefs)
        small_prefs_mask = prefs < np.finfo(np.float64).tiny
    
    avg_entropy = 0
    for i in range(num):
        avg_entropy += dirichlet.entropy(prefs[i, :]) / num
        if np.isnan(avg_entropy):
            x = "What?"
    return avg_entropy        

class AverageUserEntropy(Measurement):
    """
    Calculates the average dirichlet entropy across all users
    """
    def __init__(self, name="avg_user_entropy", verbose=False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender, **kwargs):
        actual_profiles = recommender.actual_user_profiles
        avg_user_entropy = avg_entropy(actual_profiles)
        self.observe(avg_user_entropy)

class AveragePredictedUserEntropy(Measurement):
    """
    Calculates the average dirichlet entropy across all users
    """
    def __init__(self, name="avg_predicted_user_entropy", verbose=False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender, **kwargs):
        predicted_profiles = recommender.predicted_user_profiles
        avg_user_entropy = avg_entropy(predicted_profiles)
        self.observe(avg_user_entropy)