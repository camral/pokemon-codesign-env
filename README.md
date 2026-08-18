# Pokemon Showdown Codesign Environment

**Pokemon Showdown** - A turn-based Pokemon battle simulator. This is based off [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) and [Poke-Env](https://github.com/hsahovic/poke-env).
In this problem setup, the agent is tasked with designing a Pokemon team (embodiment) and learning a battle strategy (control policy) to maximize its win rate against various opponents.

The difficulties of the learning problems are yet to be fully explored, but the environment is designed to be flexible and allow for a wide range of co-design challenges.
Potential environment setups:
 - Randomized opponent teams with fixed battle rules.
 - Randomized battle rules with fixed opponent teams.
 - Randomized opponent teams and battle rules.
 - Curated opponent teams and battle rules, designed to test specific co-design strategies.
 - Self-play, where the agent competes against its own previous versions to iteratively improve both its team design and battle strategy.
 - Multi-agent co-design, where multiple agents design their own teams and strategies, competing against each other in a shared environment.