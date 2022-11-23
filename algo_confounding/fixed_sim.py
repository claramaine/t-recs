# import trecs
import numpy as np
from scipy.spatial.distance import pdist

from trecs.components import ExtendedUsers, Items, Creators, ActualUserScores
from trecs.models import ContentFiltering, PopularityRecommender, ImplicitMF, SocialFiltering
from trecs.metrics import Measurement, RMSEMeasurement, InteractionSpread, AverageFeatureScoreRange, InteractionSimilarity
from trecs.random import Generator
from collections import defaultdict
from chaney_utils import (
    gen_social_network,
    mu_sigma_to_alpha_beta,
    exclude_new_items,
    perfect_scores,
    interleave_new_items,
    process_measurement,
    MeanDistanceSimUsers,
    MeanInteractionDistance,
    SimilarUserInteractionSimilarity,
    InteractionTracker,
    RandomRecommender,
    ChaneyContent,
    IdealRecommender,
    TargetIdealRecommender
)
from replication_sim import NewItemFactory #thesis:new needed for the fixed catalog implementation
from udpf_utils import (
    PredictedTargetSimilarity, 
    ActualTargetSimilarity, 
    MostPrefferedAttributeChosen, 
    ActualUserProfiles,
    TargetUserProfiles,
    AverageUserEntropy,
    AveragePredictedUserEntropy,
    PreferenceChangeL2Norm,
    PreferenceChangeTVD
)
import argparse
import os
import errno
import warnings
import pprint
import pickle as pkl
warnings.simplefilter("ignore")
        


# generates user scores on the fly
def ideal_content_score_fns(sigma, mu_n, num_items_per_iter, generator):
    """
    This is the scoring function used for the Ideal Recommender when content creators are introduced.
    This is necessary because scores are generated for items on the fly, rather than being generated
    at the beginning of the simulation.

    Returns the score function for the model and the score function for the users.
    """
    alpha, beta = mu_sigma_to_alpha_beta(mu_n, sigma)
    # start with empty array
    true_user_item_utils = None
    def model_score_fn(user_attrs, item_attrs):
        # generate utils for new items
        nonlocal true_user_item_utils
        num_users, num_items = user_attrs.shape[0], item_attrs.shape[1]
        if true_user_item_utils is None:
            # initialize empty user item score array
            true_user_item_utils = np.array([]).reshape((num_users, num_items))
        if item_attrs.shape[1] == num_items_per_iter: # EDGE CASE: when num_items_per_iter = num_items
            true_utils_mean = user_attrs @ item_attrs
            true_utils_mean = np.clip(true_utils_mean, 1e-9, None)
            user_alphas, user_betas = mu_sigma_to_alpha_beta(true_utils_mean, sigma)
            true_utility = generator.beta(user_alphas, user_betas, size=(num_users, num_items))
            true_user_item_utils = np.hstack((true_user_item_utils, true_utility)) #TODO: why a stack??
            # all predicted scores for these "new" items will be negative infinity,
            # ensuring they never get recommended
            return np.ones(true_user_item_utils.shape) * float('-inf')
        else:
            # assume this is when train() was called on the entire item set
            assert (num_users, num_items) == true_user_item_utils.shape
            return true_user_item_utils[:num_users, :num_items].copy() # subset to correct dimensions

    def user_score_fn(user_profiles, item_attributes):
        """
        The function that calculates user scores depends on the true utilities,
        which are generated during the `model_score_fn` step.
        """
        nonlocal alpha
        nonlocal beta
        num_items = item_attributes.shape[1]
        num_users = user_profiles.shape[0]
        # calculate percentage of utility known
        perc_util_known = generator.beta(alpha, beta, size=(num_users, num_items))
        # get the true utilities for the items that were just created;
        # they should be the most recently added items to the user-item
        # score matrix - this is only true for the ideal recommender
        true_utility = true_user_item_utils[:, -num_items:]
        known_utility = true_utility * perc_util_known
        return known_utility

    return model_score_fn, user_score_fn

def user_score_fn(rec, mu_n, sigma, generator):
    alpha, beta = mu_sigma_to_alpha_beta(mu_n, sigma)
    def score_fn(user_profiles, item_attributes):
        # generator = Generator(seed=1234) # thesis:TODO only a test pls remove
        nonlocal alpha
        nonlocal beta
        num_items = item_attributes.shape[1]
        num_users = user_profiles.shape[0]
        # calculate percentage of utility known
        perc_util_known = generator.beta(alpha, beta, size=(num_users, num_items))
        true_utils_mean = user_profiles @ item_attributes
        true_utils_mean = np.clip(true_utils_mean, 1e-9, None)
        user_alphas, user_betas = mu_sigma_to_alpha_beta(true_utils_mean, sigma)
        true_utility = generator.beta(user_alphas, user_betas, size=(num_users, num_items))
        known_utility = true_utility * perc_util_known
        return known_utility

    return score_fn

def sample_users_and_items(rng, num_users, num_items, num_attrs, num_sims):
    # multiplier of *10 comes from Chaney paper
    user_params = rng.dirichlet(np.ones(num_attrs), size=num_sims) * 10
        #user_params is an array of floats for each of the 20 attributes
        #this is a global vector of popularity over all users
    user_params_target = rng.dirichlet(np.ones(num_attrs), size=num_sims) * 10
        # and this is a global vector of target popularity over all users (?) thesis:TODO make this better
    item_params = rng.dirichlet(np.ones(num_attrs) * 100, size=num_sims) * 0.1


    # each element in users is the users vector in one simulation
    users, user_targets, items, social_networks = [], [], [], [] #thesis:changed
    for sim_index in range(num_sims):
        # generate user preferences and item attributes
        user_prefs = rng.dirichlet(user_params[sim_index, :], size=num_users) # 100 users
            #100 users x 20 attr each
        item_attrs = rng.dirichlet(item_params[sim_index, :], size=num_items) # 200 items
            
        # thesis:changed
        user_prefs_target = rng.dirichlet(user_params_target[sim_index, :], size=num_users)

        # add all synthetic data to list
        users.append(user_prefs)
        user_targets.append(user_prefs_target) # thesis:new
        social_networks.append(gen_social_network(user_prefs))
        items.append(item_attrs)


    return users, user_targets, items, social_networks

def init_sim_state(user_profiles, target_user_profiles, all_items, args, rng): # thesis:changed
    # each user interacts with items based on their (noisy) knowledge of their own scores
    # user choices also depend on the order of items they are recommended
    u = ExtendedUsers(
        actual_user_profiles=user_profiles,
        target_user_profiles=target_user_profiles, # thesis:new
        size=(args["num_users"], args["num_attrs"]),
        num_users=args["num_users"],
        attention_exp=args["attention_exp"],
        repeat_interactions=False,
        drift=args["drift"]
    )
    c = NewItemFactory(np.copy(all_items), args["new_items_per_iter"])
    empty_item_set = np.array([]).reshape((args["num_attrs"], 0)) # initialize empty item set

    init_params = {
        "num_items_per_iter": args["new_items_per_iter"],
        "num_users": args["num_users"],
        "num_items": 0, # all simulations start with 0 items
        "interleaving_fn": interleave_new_items(rng),
    }

    run_params = {
        "random_items_per_iter": args["new_items_per_iter"],
        "vary_random_items_per_iter": False, # exactly X items are interleaved
    }

    return u, c, empty_item_set, init_params, run_params

def calc_startup_rec_size(args):
    """
    Helper function for determining the number of items to recommend per iteration after startup
    """
    if args["repeated_training"]:
        post_startup_rec_size = "all"
    else: # only serve items that were in the initial training set
        post_startup_rec_size = (args["startup_iters"] + 1) * args["new_items_per_iter"] # show all items from training set plus interleaved items
    return post_startup_rec_size

def construct_metrics(keys, **kwargs):
    """
    Helper function for constructing metrics to measure relative to the ideal
    """
    metrics = []
    for k in keys:
        if k == "mean_item_dist": # mean interaction distance
            metrics.append(MeanInteractionDistance(kwargs["pairs"], name="mean_item_dist"))
        elif k == "sim_user_dist": # mean distance similar users
            metrics.append(MeanDistanceSimUsers(kwargs["ideal_interactions"], kwargs["ideal_item_attrs"]))
        elif k == "interaction_history": # interaction tracker
            metrics.append(InteractionTracker())
        elif k == "rmse": # error measurement
            metrics.append(RMSEMeasurement())
        # thesis:new
        elif k == "predicted_target_similarity": # similarity of u_hat and u_target
            metrics.append(PredictedTargetSimilarity(kwargs["target_prefs"]))
        elif k == "actual_target_similarity": # similarity of u and u_target
            metrics.append(ActualTargetSimilarity(kwargs["target_prefs"]))
        
        elif k == "actual_user_profiles": 
            metrics.append(ActualUserProfiles())
        elif k == "target_user_profiles": 
            metrics.append(TargetUserProfiles())
        
        elif k == "most_preffered_attribute_chosen":
            metrics.append(MostPrefferedAttributeChosen())
        elif k == "avg_user_entropy":
            metrics.append(AverageUserEntropy())
        elif k == "avg_pred_user_entropy":
            metrics.append(AveragePredictedUserEntropy())
            
        elif k == "interaction_spread": # diversity
            metrics.append(InteractionSpread(kwargs["pairs"]))
        elif k == "afsr": # alt diversity
            metrics.append(AverageFeatureScoreRange())
        elif k == "interaction_similarity": # chaney homog
            metrics.append(InteractionSimilarity(kwargs["pairs"]))
        elif k == "pref_change_l2":
            metrics.append(PreferenceChangeL2Norm())

    return metrics

"""
Methods for running each type of simulation
"""
def run_ideal_sim(user_prefs, target_prefs, item_attrs, metrics, args, rng): #thesis:changed
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, target_prefs, item_attrs, args, rng) #thesis:changed
    post_startup_rec_size = calc_startup_rec_size(args)
    model_score_fn, user_score_fn = ideal_content_score_fns(args["sigma"], args["mu_n"], args["new_items_per_iter"], rng)
    ideal = IdealRecommender(
        user_representation=user_prefs,
        creators=c,
        actual_user_representation=u,
        actual_item_representation=empty_items,
        score_fn=model_score_fn,
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    ideal.users.set_score_function(user_score_fn)
    ideal.add_metrics(*metrics)
    ideal.startup_and_train(timesteps=args["startup_iters"])
    ideal.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved items
    ideal.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], **run_params)
    ideal.close()
    return ideal

# thesis:new
def run_target_ideal_sim(user_prefs, target_prefs, item_attrs, metrics, args, rng):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, target_prefs, item_attrs, args, rng) #TODO: new
    post_startup_rec_size = calc_startup_rec_size(args)
    model_score_fn, user_score_fn = ideal_content_score_fns(args["sigma"], args["mu_n"], args["new_items_per_iter"], rng)
    ideal = TargetIdealRecommender(
        user_representation=target_prefs.copy(), #thesis:TODO if target_prefs are nonstationary does it make it bad? -> i guess just update it below for each run of the sim?
        creators=c,
        actual_user_representation=u,
        actual_item_representation=empty_items,
        score_fn=model_score_fn,
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    ideal.users.set_score_function(user_score_fn)
    ideal.add_metrics(*metrics)
    ideal.startup_and_train(timesteps=args["startup_iters"])
    ideal.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved items
    ideal.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], **run_params)
    ideal.close()
    return ideal

def run_content_sim(user_prefs, target_prefs, item_attrs, metrics, args, rng, judgements_on=False):
    fresh_rng = rng # Generator(seed=1234) # thesis:TODO
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, target_prefs, item_attrs, args, fresh_rng) # thesis:changed added target and changed rng
    post_startup_rec_size = calc_startup_rec_size(args)
    chaney = ChaneyContent(
        creators=c,
        num_attributes=args["num_attrs"],
        actual_item_representation=empty_items,
        actual_user_representation=u,
        score_fn=exclude_new_items(args["new_items_per_iter"]),
        judgements_on=judgements_on, # thesis:new
        j_strength=args["alpha"], # thesis:new
        seed=args["seed"], # thesis:new only added to content for purposes of comparison
        **init_params)
    # set score function here because it requires a reference to the recsys
    chaney.users.set_score_function(user_score_fn(chaney, args["mu_n"], args["sigma"], fresh_rng))
    chaney.add_metrics(*metrics)
    chaney.add_state_variable(chaney.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    chaney.startup_and_train(timesteps=args["startup_iters"]) # update user representations, but only serve random items
    chaney.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved items
    chaney.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], **run_params)
    chaney.close() # end logging
    return chaney

def run_mf_sim(user_prefs, target_prefs, item_attrs, metrics, args, rng, judgements_on=False):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, target_prefs, item_attrs, args, rng) #TODO: new
    post_startup_rec_size = calc_startup_rec_size(args)
    mf = ImplicitMF(
        creators=c,
        actual_item_representation=empty_items,
        actual_user_representation=u,
        num_latent_factors=args["num_attrs"],
        score_fn=exclude_new_items(args["new_items_per_iter"]),
        judgements_on=judgements_on, # thesis:new
        j_strength=args["alpha"],
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    mf.users.set_score_function(user_score_fn(mf, args["mu_n"], args["sigma"], rng))
    mf.add_metrics(*metrics)
    mf.add_state_variable(mf.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    mf.startup_and_train(timesteps=args["startup_iters"], no_new_items=False) # update user representations, but only serve random items
    mf.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved items
    mf.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], reset_interactions=False, **run_params)
    mf.close() # end logging
    return mf

def run_sf_sim(social_network, user_prefs, target_prefs, item_attrs, metrics, args, rng):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, target_prefs, item_attrs, args, rng) #TODO: new
    post_startup_rec_size = calc_startup_rec_size(args)
    sf = SocialFiltering(
        creators=c,
        user_representation=social_network,
        actual_item_representation=empty_items,
        actual_user_representation=u,
        score_fn=exclude_new_items(args["new_items_per_iter"]),
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    sf.users.set_score_function(user_score_fn(sf, args["mu_n"], args["sigma"], rng))
    sf.add_metrics(*metrics)
    sf.add_state_variable(sf.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    sf.startup_and_train(timesteps=args["startup_iters"])
    sf.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved items
    sf.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], **run_params)
    sf.close() # end logging
    return sf

def run_pop_sim(user_prefs, target_prefs, item_attrs, metrics, args, rng):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, target_prefs, item_attrs, args, rng) #TODO: new
    post_startup_rec_size = calc_startup_rec_size(args)
    p = PopularityRecommender(
        creators=c,
        actual_item_representation=empty_items,
        actual_user_representation=u,
        score_fn=exclude_new_items(args["new_items_per_iter"]),
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    p.users.set_score_function(user_score_fn(p, args["mu_n"], args["sigma"], rng))
    p.add_metrics(*metrics)
    p.add_state_variable(p.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    p.startup_and_train(timesteps=args["startup_iters"])
    p.set_num_items_per_iter(post_startup_rec_size)
    p.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], **run_params)
    p.close() # end logging
    return p

def run_random_sim(user_prefs, target_prefs, item_attrs, metrics, args, rng):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, target_prefs, item_attrs, args, rng)
    post_startup_rec_size = calc_startup_rec_size(args)
    r = RandomRecommender(
        creators=c,
        num_attributes=args["num_attrs"],
        actual_item_representation=empty_items,
        actual_user_representation=u,
        score_fn=exclude_new_items(args["new_items_per_iter"]),
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    r.users.set_score_function(user_score_fn(r, args["mu_n"], args["sigma"], rng))
    r.add_metrics(*metrics) # random pairing of users
    r.add_state_variable(r.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    r.startup_and_train(timesteps=args["startup_iters"])
    r.set_num_items_per_iter(post_startup_rec_size)
    # always train between steps so items are randomly scored
    r.run(timesteps=args["sim_iters"], train_between_steps=True, **run_params)
    r.close() # end logging
    return r

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='running simulations')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_users', type=int, default=100)
    parser.add_argument('--num_items', type=int, default=10000) #if this is too small the system runs out of items and throws an error
    parser.add_argument('--repeated_training', dest='repeated_training', action='store_true')
    parser.add_argument('--single_training', dest='repeated_training', action='store_false')
    parser.add_argument('--items_per_creator', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--mu_n', type=float, default=0.98)
    parser.add_argument('--sigma', type=float, default=1e-5)
    
    # thesis:new and changed
    profile_metrics = ["actual_user_profiles", "target_user_profiles"]
    user_homog_metrics = ["avg_user_entropy"]#, "avg_pred_user_entropy"]#, "most_preffered_attribute_chosen"]
    profile_and_homog_metrics = ["actual_user_profiles", "target_user_profiles", "avg_user_entropy", "avg_pred_user_entropy"]
    
    preference_metrics = ["predicted_target_similarity", "actual_target_similarity"]
    profile_and_preference_metrics = ["actual_user_profiles", "target_user_profiles", "predicted_target_similarity", "actual_target_similarity"]
    
    profile_simiarity_diversity_homog_change_metrics = ["actual_user_profiles", "target_user_profiles", "predicted_target_similarity", "actual_target_similarity", "interaction_spread", "interaction_similarity", "pref_change_l2"]
    
    other_metrics = ["rmse", "mean_item_dist"]
    
    current_metrics = user_homog_metrics # profile_simiarity_diversity_homog_change_metrics
    ideal_model_metrics = current_metrics.copy()
    ideal_model_metrics.append("interaction_history")
    current_metrics.append("rmse")
        
    parser.add_argument('--output_dir', type=str, default="t-recs/algo_confounding/udpc_ouput_homog_fixed")
    parser.add_argument('--startup_iters', type=int, default=50)
    parser.add_argument('--sim_iters', type=int, default=350) #timesteps
    parser.add_argument('--num_sims', type=int, default=5)
    parser.add_argument('--num_attrs', type=int, default=20)
    parser.add_argument('--metrics', nargs='+', default=current_metrics)
    parser.add_argument('--judgements', dest='judgements_on', default=False, action='store_true') # thesis:new
    parser.add_argument('--alpha', type=float, default=1.0)
    
    # factors maybe important to homog
    parser.add_argument('--attention_exp', type=float, default=-0.8) #old default was -0.8
    parser.add_argument('--new_items_per_iter', type=int, default=10) #old default was 10
    parser.add_argument('--drift', type=float, default=0.02) #how big this is determines how fast the homog sets in

    parsed_args = parser.parse_args(['--repeated_training', '--judgements']) #thesis:new, needed to make this boolean true
    args = vars(parsed_args)

    print("Creating experiment output folder... 💻")
    # create output folder
    try:
        os.makedirs(args["output_dir"])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # write experiment arguments to file
    with open(os.path.join(args["output_dir"], "args.txt"), "w") as log_file:
        pprint.pprint(args, log_file)

    rng = np.random.default_rng(args["seed"])

    # sample initial user / creator profiles
    print("Sampling initial user and creator profiles... 🔬")
    users, user_targets, items, social_networks = sample_users_and_items(
        rng,
        args["num_users"],
        args["num_items"],
        args["num_attrs"],
        args["num_sims"]
    )

    # run simulations
    model_keys = ["ideal", "target_ideal", "content_chaney", "content_chaney_j_3.0", "content_chaney_j_1.0", "content_chaney_j_0.5", "random", "popularity"]
    # stores results for each type of model for each type of user pairing (random or cosine similarity)
    result_metrics = {k: defaultdict(list) for k in args['metrics']}
    models = {} # temporarily stores models  

    print("Running simulations...👟")
    for i in range(args["num_sims"]):
        print("Runnning Sim", i)
        true_prefs = users[i] # underlying true preferences
        target_prefs = user_targets[i] # thesis:new
        item_representation = items[i].T
        social_network = social_networks[i]

        # generate random pairs for evaluating jaccard similarity
        pairs = [rng.choice(args["num_users"], 2, replace=False) for _ in range(800)]
        
        print("ideal")
        metrics = construct_metrics(ideal_model_metrics, pairs=pairs, target_prefs=target_prefs)
        models["ideal"] = run_ideal_sim(true_prefs, target_prefs, item_representation, metrics, args, rng)
        ideal_interactions = np.hstack(process_measurement(models["ideal"], "interaction_history")) # pull out the interaction history for the ideal simulations
        ideal_attrs = models["ideal"].actual_item_attributes
        
        print("content Ms")
        if args["judgements_on"] == True:
            args["alpha"] = 3.0
            # print('\talpha=', args['alpha'])
            metrics = construct_metrics(args['metrics'], pairs=pairs, ideal_interactions=ideal_interactions, ideal_item_attrs=ideal_attrs, target_prefs=target_prefs)
            models["content_chaney_j_3.0"] = run_content_sim(true_prefs, target_prefs, item_representation, metrics, args, rng, judgements_on=True)
            
            args["alpha"] = 1.0
            # print('\talpha=', args['alpha'])
            metrics = construct_metrics(args['metrics'], pairs=pairs, ideal_interactions=ideal_interactions, ideal_item_attrs=ideal_attrs, target_prefs=target_prefs)
            models["content_chaney_j_1.0"] = run_content_sim(true_prefs, target_prefs, item_representation, metrics, args, rng, judgements_on=True)
            
            args["alpha"] = 0.5
            # print('\talpha=', args['alpha'])
            metrics = construct_metrics(args['metrics'], pairs=pairs, ideal_interactions=ideal_interactions, ideal_item_attrs=ideal_attrs, target_prefs=target_prefs)
            models["content_chaney_j_0.5"] = run_content_sim(true_prefs, target_prefs, item_representation, metrics, args, rng, judgements_on=True)

        print("content basic")
        metrics = construct_metrics(args['metrics'], pairs=pairs, ideal_interactions=ideal_interactions, ideal_item_attrs=ideal_attrs, target_prefs=target_prefs)
        models["content_chaney"] = run_content_sim(true_prefs, target_prefs, item_representation, metrics, args, rng, judgements_on=False)

        print("target ideal")
        metrics = construct_metrics(ideal_model_metrics, pairs=pairs, target_prefs=target_prefs)
        models["target_ideal"] = run_target_ideal_sim(true_prefs, target_prefs, item_representation, metrics, args, rng)
        # thesis:TODO maybe have to do something like the above? would have to add on another ideal_interactions param to a bunch of code below

        print("random")
        metrics = construct_metrics(args['metrics'], pairs=pairs, ideal_interactions=ideal_interactions, ideal_item_attrs=ideal_attrs, target_prefs=target_prefs)
        models["random"] = run_random_sim(true_prefs, target_prefs, item_representation, metrics, args, rng)

        print("popularity")        
        metrics = construct_metrics(args['metrics'], pairs=pairs, ideal_interactions=ideal_interactions, ideal_item_attrs=ideal_attrs, target_prefs=target_prefs)
        models["popularity"] = run_pop_sim(true_prefs, target_prefs, item_representation, metrics, args, rng)
        
        # extract results from each model
        for model_key in model_keys:
            model = models[model_key]
            metric_keys = model.get_measurements().keys()
            for metric_key in metric_keys:
                if metric_key in result_metrics: # only record metrics specified by the user
                    result_metrics[metric_key][model_key].append(process_measurement(model, metric_key))

    # write results to pickle file
    output_file = os.path.join(args["output_dir"], "sim_results.pkl")
    pkl.dump(result_metrics, open(output_file, "wb"), -1)
    print("Done with simulation! 🎉")