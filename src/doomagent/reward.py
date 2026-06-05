from typing import Dict, Tuple

from doom_arena.reward import VizDoomReward


class CustomReward(VizDoomReward):
    """
    Drop-in replacement for the grading reward. Modify __call__ to experiment
    with reward shaping without changing the environment.

    The grading server uses VizDoomReward directly, so changes here only
    affect training. sum(returned_tuple) is the scalar reward the agent sees.
    """

    def __call__(
        self,
        vizdoom_reward: float,
        game_var: Dict[str, float],
        game_var_old: Dict[str, float],
        player_id: int,
    ) -> Tuple:
        # Base rewards: hits +2, hits_taken -0.1, frags +100
        base = super().__call__(vizdoom_reward, game_var, game_var_old, player_id)

        # Suggested additions — uncomment and tune as needed:
        # rwd_health = 0.01 * (game_var["HEALTH"] - game_var_old["HEALTH"])
        # rwd_ammo   = 0.05 * (game_var["AMMO3"]  - game_var_old["AMMO3"])

        return base


class DeathPenaltyReward(VizDoomReward):
    """Base reward + penalty on death to discourage passive/suicidal behaviour."""

    def __init__(self, num_players: int, death_penalty: float = 10.0):
        super().__init__(num_players)
        self.death_penalty = death_penalty

    def __call__(
        self,
        vizdoom_reward: float,
        game_var: Dict[str, float],
        game_var_old: Dict[str, float],
        player_id: int,
    ) -> Tuple:
        base = super().__call__(vizdoom_reward, game_var, game_var_old, player_id)
        rwd_dead = -self.death_penalty if game_var["DEAD"] > game_var_old["DEAD"] else 0.0
        return (*base, rwd_dead)
