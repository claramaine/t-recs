"""
BaseRecommender, the foundational class for all recommender systems
implementable in our simulation library
"""
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from trecs.metrics import MeasurementModule
from trecs.components import (
    Users,
    Items,
    PredictedScores,
    PredictedUserProfiles,
    SystemStateModule,
)
from trecs.logging import VerboseMode
from trecs.matrix_ops import inner_product
from trecs.random import Generator
from trecs.utils import is_valid_or_none


class BaseRecommender(MeasurementModule, SystemStateModule, VerboseMode, ABC):
    """Abstract class representing a recommender system.

    The attributes and methods in this class can be generalized beyond
    recommender systems and are currently common to all pre-loaded models.

    Parameters
    -----------

        users_hat: :obj:`numpy.ndarray`
            An array representing users. The shape and meaning depends on
            the implementation of the concrete class.

        items_hat: :obj:`numpy.ndarray`
            An array representing items. The shape and meaning depends on
            the implementation of the concrete class.

        users: :obj:`numpy.ndarray` or :class:`~components.users.Users`
            An array representing real user preferences unknown to the
            system. Shape is :math:`|U| \\times |A|`, where :math:`|A|` is the
            number of attributes and :math:`|U|` is the number of users. When
            a `numpy.ndarray` is passed in, we assume this represents the user
            *scores*, not the users' actual attribute vectors.

        items: :obj:`numpy.ndarray` or :class:`~components.items.Items`
            An array representing real item attributes unknown to the
            system. Shape is :math:`|A|\\times|I|`, where :math:`|I|` is the
            number of items and :math:`|A|` is the number of attributes.

        num_users: int
            The number of users in the system.

        num_items: int
            The number of items in the system.

        num_items_per_iter: int
            Number of items presented to the user at each iteration.

        measurements: list
            List of metrics to monitor.

        record_base_state: bool (optional, default: False)
            If True, the system will record at each time step its internal
            representation of users profiles and item profiles, as well as the
            true user profiles and item profiles. It will also record the
            predicted user-item scores at each time step.

        system_state: list
            List of system state components to monitor.

        verbose: bool (optional, default: False)
            If True, it enables verbose mode.

        seed: int, None (optional, default: None)
            Seed for random generator used

    Attributes
    -----------

        users_hat: :class:`~components.users.PredictedUserProfiles`
            An array representing users, matching user_representation. The
            shape and meaning depends on the implementation of the concrete
            class.

        items_hat: :class:`~components.items.Items`
            An array representing items, matching item_representation. The
            shape and meaning depends on the implementation of the concrete
            class.

        users: :class:`~components.users.Users`
            An array representing real user preferences. Shape should be
            :math:`|U| \\times |A|`, and should match items.

        items: :class:`~components.items.Items`
            An array representing actual item attributes. Shape should be
            :math:`|A| \\times |I|`, and should match users.

        predicted_scores: :class:`~components.users.PredictedScores`
            An array representing the user preferences as perceived by the
            system. The shape is always :math:`|U| \\times |I|`, where
            :math:`|U|` is the number of users in the system and :math:`|I|`
            is the number of items in the system. The scores are calculated with
            the dot product of :attr:`users_hat` and :attr:`items_hat`.

        num_users: int
            The number of users in the system.

        num_items: int
            The number of items in the system.

        num_items_per_iter: int
            Number of items presented to the user per iteration.

        probabilistic_recommendations: bool (optional, default: False)
            When this flag is set to `True`, the recommendations (excluding
            any random interleaving) will be randomized, meaning that items
            will be recommended with a probability proportionate to their
            predicted score, rather than the top `k` items, as ranked by their
            predicted score, being recommended.

        random_state: :class:`trecs.random.generators.Generator`

        indices: :obj:`numpy.ndarray`
            A :math:`|U| \\times |I|` array representing the past interactions of each
            user. This keeps track of which items each user has interacted
            with, so that it won't be presented to the user again if
            `repeated_items` are not allowed.

        score_fn: callable
            Function that is used to calculate each user's scores for each
            candidate item. Note that this function can be the same function
            used by the recommender system to generate its predictions for
            user-item scores. The score function should take as input
            user_profiles and item_attributes.
    """

    @abstractmethod
    def __init__(  # pylint: disable=R0913,R0912,R0915
        self,
        users_hat,
        items_hat,
        users,
        items,
        num_users,
        num_items,
        num_items_per_iter,
        probabilistic_recommendations=False,
        measurements=None,
        record_base_state=False,
        system_state=None,
        verbose=False,
        score_fn=inner_product,
        seed=None,
    ):
        # Init logger
        VerboseMode.__init__(self, __name__.upper(), verbose)
        # Initialize measurements
        MeasurementModule.__init__(self)
        if measurements is not None:
            self.add_metrics(*measurements)
        # init the recommender system's internal representation of users
        # and items
        self.users_hat = PredictedUserProfiles(users_hat)
        self.items_hat = Items(items_hat)
        assert callable(score_fn)  # score function must be a function
        self.score_fn = score_fn
        # set predicted scores
        self.predicted_scores = None
        self.update_predicted_scores()
        assert self.predicted_scores is not None
        # determine whether recommendations should be randomized, rather than
        # top-k by predicted score
        self.probabilistic_recommendations = probabilistic_recommendations

        if not is_valid_or_none(num_users, int):
            raise TypeError("num_users must be an int")
        if not is_valid_or_none(num_items, int):
            raise TypeError("num_items must be an int")
        if not is_valid_or_none(num_items_per_iter, int):
            raise TypeError("num_items_per_iter must be an int")
        assert num_items_per_iter > 0  # check number of items per iteration is positive
        if not hasattr(self, "metrics"):
            raise ValueError("You must define at least one measurement module")

        # check users array
        if not is_valid_or_none(users, (list, np.ndarray, Users)):
            raise TypeError("users must be array_like or Users")
        if users is None:
            self.users = Users(size=self.users_hat.shape, num_users=num_users, seed=seed)
        if isinstance(users, (list, np.ndarray)):
            # assume that's what passed in is the user's profiles
            self.users = Users(actual_user_profiles=users, num_users=num_users)
        if isinstance(users, Users):
            self.users = users

        # check items array
        if not is_valid_or_none(items, (list, np.ndarray, Items)):
            raise TypeError("items must be array_like or Items")
        if items is None:
            raise ValueError("true item attributes can't be None")
        if isinstance(items, (list, np.ndarray)):
            # will need to change this when Items no longer inherits from
            # ndarray
            self.items = Items(items)
        if isinstance(items, Items):
            self.items = items

        # system state
        SystemStateModule.__init__(self)
        if record_base_state:
            self.add_state_variable(
                self.users_hat,
                self.users,
                self.items_hat,
                self.predicted_scores,
            )
        if system_state is not None:
            self.add_state_variable(*system_state)

        # initialize actual user scores for items
        self.users.set_score_function(score_fn)
        # users compute their own scores using the true item attributes,
        # unless their own scores are already known to them
        if self.users.get_actual_user_scores() is None:
            self.users.compute_user_scores(self.items)

        assert self.users and isinstance(self.users, Users)
        self.num_users = num_users
        self.num_items = num_items
        self.num_items_per_iter = num_items_per_iter
        self.random_state = Generator(seed)
        # Matrix keeping track of the items consumed by each user
        self.indices = np.tile(np.arange(num_items), (num_users, 1))
        self.log("Recommender system ready")
        self.log("Num items: %d" % self.num_items)
        self.log("Users: %d" % self.num_users)
        self.log("Items per iter: %d" % self.num_items_per_iter)
        if seed is not None:
            self.log("Set seed to %d" % seed)
        else:
            self.log("Seed was not set.")

    def __del__(self):
        """
        Closes the logging handler upon garbage collection.
        """
        self.close()

    def update_predicted_scores(self):
        """
        Updates scores predicted by the system based on past interactions for
        better user predictions. Specifically, it updates :attr:`predicted_scores`
        with a dot product.

        Returns
        --------
            predicted_scores: :class:`~components.users.PredictedScores`
        """
        user_profiles = self.users_hat
        item_attributes = self.items_hat
        predicted_scores = self.score_fn(user_profiles, item_attributes)
        self.log(
            "System updates predicted scores given by users (rows) "
            + "to items (columns):\n"
            + str(predicted_scores)
        )
        assert predicted_scores is not None
        if self.predicted_scores is None:
            self.predicted_scores = PredictedScores(predicted_scores)
        else:
            self.predicted_scores[:, :] = predicted_scores

    def generate_recommendations(self, k=1, item_indices=None):
        """
        Generate recommendations for each user.

        Parameters
        -----------

            k : int (optional, default: 1)
                Number of items to recommend.

            item_indices : :obj:`numpy.ndarray` or None (optional, default: None)
                A matrix containing the indices of the items each user has not yet
                interacted with. It is used to ensure that the user is presented
                with items they have not already interacted with. If `None`,
                then the user may be recommended items that they have already
                interacted with.

        Returns
        ---------
            Recommendations: :obj:`numpy.ndarray`
        """
        if item_indices is not None:
            if item_indices.size < self.num_users:
                raise ValueError(
                    "At least one user has interacted with all items!"
                    "To avoid this problem, you may want to allow repeated items."
                )
            if k > item_indices.shape[1]:
                raise ValueError(
                    f"There are not enough items left to recommend {k} items to each user."
                )
        row = np.repeat(self.users.user_vector, item_indices.shape[1])
        row = row.reshape((self.num_users, -1))
        self.log("Row:\n" + str(row))
        self.log("Item indices:\n" + str(item_indices))
        s_filtered = self.predicted_scores[row, item_indices]
        # scores are U x I; we can use argsort to sort the item indices
        # from low to high scores
        permutation = s_filtered.argsort()
        rec = item_indices[row, permutation]
        self.log("Items ordered by preference (low to high) for each user:\n" + str(rec))
        if self.probabilistic_recommendations:
            # the recommended items will not be exactly determined by
            # predicted score; instead, we will sample from the sorted list
            # such that higher-preference items get more probability mass
            num_items_unseen = rec.shape[1]  # number of items unseen per user
            probabilities = np.logspace(0.0, num_items_unseen / 10.0, num=num_items_unseen, base=2)
            probabilities = probabilities / probabilities.sum()
            picks = np.random.choice(num_items_unseen, k, replace=False, p=probabilities)
            return rec[:, picks]
        else:
            return rec[:, -k:]

    def recommend(
        self,
        startup=False,
        random_items_per_iter=0,
        vary_random_items_per_iter=False,
        repeated_items=True,
    ):
        """
        Implements the recommendation process by combining recommendations and
        new (random) items.

        Parameters
        -----------
            startup: bool (optional, default: False)
                If True, the system is in "startup"  (exploration) mode and
                only presents the user with new randomly chosen items. This is
                done to maximize exploration.

            random_items_per_iter: int (optional, default: 0)
                Number of per-user item recommendations that should be
                randomly generated. Passing in `1.0` will result in all
                recommendations being randomly generated, while passing in `0.0`
                will result in all recommendations coming from predicted score.

            vary_random_items_per_iter: bool (optional, default: False)
                If true, then at each timestep, the # of items that are recommended
                randomly is itself randomly generated between 0 and
                `random_items_per_iter`, inclusive.

            repeated_items : bool (optional, default: True)
                If True, repeated items are allowed in the system -- that is,
                users can interact with the same item more than once.

        Returns
        --------
            Items: :obj:`numpy.ndarray`
                New and recommended items in random order.
        """
        if random_items_per_iter > self.num_items_per_iter:
            raise ValueError(
                "Cannot show more random items per iteration than the total number"
                " of items shown per iteration"
            )

        if startup:
            num_new_items = self.num_items_per_iter
            num_recommended = 0
        else:
            num_new_items = random_items_per_iter
            if vary_random_items_per_iter:
                num_new_items = self.random_state.integers(0, random_items_per_iter + 1)
            num_recommended = self.num_items_per_iter - num_new_items

        item_indices = self.indices
        if not repeated_items:
            # for each user, eliminate items that have been interacted with
            item_indices = item_indices[np.where(item_indices >= 0)]
            item_indices = item_indices.reshape((self.num_users, -1))

        if num_recommended > 0:
            recommended = self.generate_recommendations(
                k=num_recommended, item_indices=item_indices
            )
            self.log(
                "System recommended these items (cols) to each user "
                + "(rows):\n"
                + str(recommended)
            )
        else:
            recommended = None

        self.log("Choice among %d items" % (item_indices.shape[0]))
        if item_indices.shape[1] < num_new_items:
            self.log("Insufficient number of items left!")

        if num_new_items:
            # no guarantees that randomly interleaved items do not overlap
            # with recommended items
            col = self.random_state.integers(
                item_indices.shape[1], size=(self.num_users, num_new_items)
            )
            row = np.repeat(self.users.user_vector, num_new_items).reshape((self.num_users, -1))
            new_items = item_indices[row, col]
            self.log(
                "System picked these items (cols) randomly for each user "
                + "(rows):\n"
                + str(new_items)
            )

        if num_recommended and num_new_items:
            items = np.concatenate((recommended, new_items), axis=1)
        elif num_new_items:
            items = new_items
        else:
            items = recommended
        self.random_state.shuffle(items.T)
        return items

    @abstractmethod
    def _update_user_profiles(self, interactions):
        """
        Updates user profiles based on last interaction.

        It must be defined in the concrete class.
        """

    def run(
        self,
        timesteps=50,
        startup=False,
        train_between_steps=True,
        random_items_per_iter=0,
        vary_random_items_per_iter=False,
        repeated_items=True,
    ):
        """
        Runs simulation for the given timesteps.

        Parameters
        -----------

            timestep : int (optional, default: 50)
                Number of timesteps for simulation.

            startup : bool (optional, default: False)
                If True, it runs the simulation in startup mode (see
                :func:`recommend` and :func:`startup_and_train`)

            train_between_steps : bool (optional, default: True)
                If True, the model is retrained after each timestep with the
                information gathered in the previous step.

            random_items_per_iter: float (optional, default: 0)
                Percentage of per-user item recommendations that should be
                randomly generated. Passing in `1.0` will result in all
                recommendations being randomly generated, while passing in `0.0`
                will result in all recommendations coming from predicted score.

            vary_random_items_per_iter: bool (optional, default: False)
                If true, then at each timestep, the # of items that are recommended
                randomly is itself randomly generated between 0 and
                `random_items_per_iter`, inclusive.

            repeated_items : bool (optional, default: True)
                If True, repeated items are allowed in the system -- that is,
                users can interact with the same item more than once.
        """
        if not startup:
            self.log("Run -- interleave recommendations and random items " + "from now on")
        for timestep in tqdm(range(timesteps)):
            self.log("Step %d" % timestep)
            item_idxs = self.recommend(
                startup=startup,
                random_items_per_iter=random_items_per_iter,
                vary_random_items_per_iter=vary_random_items_per_iter,
                repeated_items=repeated_items,
            )
            # important: we use the true item attributes to get user feedback
            interactions = self.users.get_user_feedback(
                items_shown=item_idxs, item_attributes=self.items
            )
            if not repeated_items:
                self.indices[self.users.user_vector, interactions] = -1
            self._update_user_profiles(interactions)
            self.log(
                "System updates user profiles based on last interaction:\n" + str(self.users_hat)
            )
            # train between steps:
            if train_between_steps:
                self.update_predicted_scores()
            self.measure_content(interactions, item_idxs, step=timestep)
        # If no training in between steps, train at the end:
        if not train_between_steps:
            self.update_predicted_scores()
            self.measure_content(interactions, item_idxs, step=timesteps)

    def startup_and_train(self, timesteps=50):
        """
        Runs simulation in startup mode by calling :func:`run` with
        startup=True. For more information about startup mode, see :func:`run`
        and :func:`recommend`.

        Parameters
        -----------

            timestep : int (optional, default: 50)
                Number of timesteps for simulation
        """
        self.log("Startup -- recommend random items")
        return self.run(timesteps, startup=True, train_between_steps=False)

    def _expand_items(self):  # pylint: disable=no-self-use
        """
        Increases number of items in the system.

        Parameters
        -----------
            num_new_items (int, optional): number of new items to add to the
                system. If None, it is equal to twice the number of items
                presented to the user in one iteration.
        if num_new_items is None:
            num_new_items = 2 * self.num_items_per_iter
        if not isinstance(num_new_items, int):
            raise TypeError("num_new_items should be int, is instead %s" % (
                                                                    type(num_new_items)))
        if num_new_items < 1:
            raise ValueError("Can't increment items by a number smaller than 1!")
        #new_indices = np.tile(self.items_hat.expand_items(self, num_new_items),
        #    (self.num_users,1))
        self.indices = np.concatenate((self.indices, new_indices), axis=1)
        self.users.compute_user_scores(self.train)
        self.predicted_scores = self.train(self.users_hat, self.items_hat)
        """

    def get_measurements(self):
        """
        Returns all available measurements. For more details, please see the
        :class:`~metrics.measurement.Measurement` class.

        Returns
        --------
        Monitored measurements: dict
        """
        if len(self.metrics) < 1:
            raise ValueError("No measurement module defined")
        measurements = dict()
        for metric in self.metrics:
            measurements = {**measurements, **metric.get_measurement()}
        if "timesteps" not in measurements:
            # pick first measurement's length for # of timesteps since they're
            # going to be the same
            elapsed = np.arange(self.metrics[0].get_timesteps())
            measurements["timesteps"] = elapsed
        return measurements

    def get_system_state(self):
        """
        Return history of system state components stored in the
        :attr:`~components.base_components.BaseComponent.state_history` of the
        components stored in :attr:`.SystemStateModule._system_state`.

        Returns
        --------
            System state: dict
        """
        if len(self._system_state) < 1:
            raise ValueError("No measurement module defined")
        state = dict()
        for component in self._system_state:
            state = {**state, **component.get_component_state()}
        if "timesteps" not in state:
            # pick first measurement's length for # of timesteps since they're
            # going to be the same
            elapsed = np.arange(self._system_state[0].get_timesteps())
            state["timesteps"] = elapsed
        # FIXME: this is needed because Users.actual_user_scores is initialized to None
        if (
            "actual_user_scores" in state
            and "timesteps" in state
            and len(state["actual_user_scores"]) > len(state["timesteps"])
        ):
            state["actual_user_scores"].pop(0)
        return state

    def measure_content(self, interactions, items_shown, step):
        """
        TODO: UPDATE DOCUMENTATION
        Calls method in the :class:`Measurements` module to record metrics.
        For more details, see the :class:`Measurements` class and its measure
        method.

        Parameters
        -----------
            interactions (:obj:`numpy.ndarray`): matrix of interactions
                per users at a given time step.

            step (int): step on which the recorded interactions refers to.
        """
        for metric in self.metrics:
            metric.measure(self, step=step, interactions=interactions, items_shown=items_shown)
        for component in self._system_state:
            component.store_state()
