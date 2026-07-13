#!/usr/bin/env python3
"""Evaluate the 2023 pretrained checkpoints in the 2023 stack.

Replacement for the model-loading part of Andrej's `agent_builder`, which
lived in his (since deleted) fork of air_hockey_challenge. Runs a tournament
game between two of the pretrained models (or the baseline agent).

Examples:
    scripts/eval_2023.py                                      # balanced vs balanced
    scripts/eval_2023.py --model1 tournament_aggressive --model2 baseline
    scripts/eval_2023.py -r                                   # with rendering (needs a display)
"""
import argparse
from os import path

from air_hockey_challenge.framework.evaluate_tournament import _run_tournament
from air_hockey_challenge.utils.tournament_agent_wrapper import (
    SimpleTournamentAgentWrapper,
)
from baseline.baseline_agent.baseline_agent import BaselineAgent

from drl_air_hockey.agents.idle_mallet import IdleMalletAgent
import drl_air_hockey.agents.single_strategy_agent as ssa
from drl_air_hockey.agents.smooth_random_agent import SmoothRandomAgent
from drl_air_hockey.utils.config import DIR_MODELS
from drl_air_hockey.utils.tournament_agent_strategies import strategy_from_str

MODELS = (
    "tournament_balanced",
    "tournament_aggressive",
    "tournament_defensive",
    "tournament_balanced_no_selfplay",
    "baseline",
    "smooth_random",
)


def make_agent(
    env_info,
    agent_id,
    model,
    idle_probability=0.0,
    idle_min_steps=5,
    idle_max_steps=25,
):
    if model == "baseline":
        agent = BaselineAgent(env_info, agent_id)
    elif model == "smooth_random":
        agent = SmoothRandomAgent(env_info, agent_id)
    else:
        # SingleStrategySpaceRAgent hardcodes BalancedAgentStrategy for its
        # action-scheme kwargs (velocity scaling, operating area). Swap in the
        # strategy matching the checkpoint before construction.
        strategy = "balanced"
        if "aggressive" in model:
            strategy = "aggressive"
        elif "defensive" in model:
            strategy = "defensive"
        ssa.BalancedAgentStrategy = lambda: strategy_from_str(strategy)
        agent = ssa.SingleStrategySpaceRAgent(
            env_info,
            agent_id=agent_id,
            model_path=path.join(DIR_MODELS, f"{model}.ckpt"),
        )
    return IdleMalletAgent(
        agent,
        idle_probability=idle_probability,
        idle_min_steps=idle_min_steps,
        idle_max_steps=idle_max_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", default="tournament_balanced", choices=MODELS)
    parser.add_argument("--model2", default="tournament_balanced", choices=MODELS)
    parser.add_argument(
        "--idle-probability1",
        type=float,
        default=0.0,
        help="Per-unpaused-step chance player 1 starts a random short pause.",
    )
    parser.add_argument(
        "--idle-probability2",
        type=float,
        default=0.0,
        help="Per-unpaused-step chance player 2 starts a random short pause.",
    )
    parser.add_argument("--idle-min-steps", type=int, default=5)
    parser.add_argument("--idle-max-steps", type=int, default=25)
    parser.add_argument("-r", "--render", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=45000, help="Steps per game (45000 = full 15 min game)")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    def agent_builder(mdp, i, **kwargs):
        agent_1 = make_agent(
            mdp.env_info, 1, args.model1, args.idle_probability1,
            args.idle_min_steps, args.idle_max_steps,
        )
        agent_2 = make_agent(
            mdp.env_info, 2, args.model2, args.idle_probability2,
            args.idle_min_steps, args.idle_max_steps,
        )
        return SimpleTournamentAgentWrapper(mdp.env_info, agent_1, agent_2)

    _run_tournament(
        log_dir="logs_2023",
        agent_builder=agent_builder,
        name_1=f"1_{args.model1}",
        name_2=f"2_{args.model2}",
        n_steps=args.steps,
        n_episodes=args.episodes,
        n_cores=1,
        quiet=False,
        render=args.render,
        interpolation_order=(
            3 if args.model1 == "baseline" else -1,
            3 if args.model2 == "baseline" else -1,
        ),
    )


if __name__ == "__main__":
    main()
