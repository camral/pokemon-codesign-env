# The codesign Gymnasium API

This environment extends the Gymnasium API with embodiment design. A standard Gymnasium
environment exposes an observation space and an action space; a codesign environment adds a third,
the **embodiment space** &Theta;, and lets you choose a point in it when you reset.

```python
import pokemon_codesign_env as pce

env, base_embodiment = pce.make("random")      # launches its own Showdown server
design = env.embodiment_space.sample()
observation, info, embodiment_rejected = env.reset(seed=0, embodiment=design)
env.close()                                    # stops the server it launched
```

## What is added

| | |
| --- | --- |
| `env.embodiment_space` | the design space &Theta;, a `gymnasium.Space` |
| `make(..., is_eval_env=, max_action_space_size=)` | evaluation mode, and a bound on the action space |
| `make()` -> `(env, base_embodiment)` | the design the environment starts from |
| `env.reset(..., embodiment=)` -> `(obs, info, embodiment_rejected)` | the design to battle with, and whether it was taken |
| `env.step(action, embodiment=)` | ignored with a warning here, see *Transition-level changes* |
| `env.validate_embodiment(e)` -> `bool` | ask Showdown whether a design can actually be built |
| `env.validate_embodiments([...])` | the same question for a whole population, batched |

`is_eval_env` marks an environment that must not have its embodiment changed during evaluation.

`base_embodiment` is the design the environment starts from, chosen once when it is created. A
reset given no design falls back to it, so returning to a known state is a matter of resetting with
it again. `make_vec` returns one per environment, in order.

`max_action_space_size` belongs to the codesign API rather than to this environment. It exists for
environments whose action space depends on the embodiment, where designing a different body changes
how many actions there are. Pokemon is not one of them: the action space is fixed by the
generation, and a team always has six switch slots whether three Pokemon are designed or six, so a
design can never change its size. Gen 1&ndash;5 use 10 actions, Gen 6 14, Gen 7 18, Gen 8 22, and
Gen 9 26.

It is still accepted, and does two things here. A format needing more actions than the bound raises.
A format needing fewer has its action space padded up to the bound, with the extra actions masked
illegal in every observation, which lets one policy span formats of different generations. The
padded indices are the later generations' mechanics, so their meaning is consistent: 10&ndash;13
mega evolve, 14&ndash;17 Z-moves, 18&ndash;21 Dynamax, 22&ndash;25 Terastallize. A Generation 4
battle simply never offers them. Leaving it unset is the normal case and pads nothing.

## The embodiment space

`embodiment_space` is a `gymnasium.spaces.Dict` whose members are exactly the dictionaries the
team compiler accepts:

```python
{"pokemon_0": {"species": "Garchomp", "nature": "Jolly",
               "moves": ["Earthquake", "Dragon Claw", "Swords Dance", "Rock Slide"],
               "evs": {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252}},
 "pokemon_1": {...}}
```

Which parameters appear is decided by `TeamSearchConfig`: only those set to `OPTIMIZED` are
designed, and the rest are resolved by the environment. `sample()` always returns a member, and
`contains()` decides membership.

Three constraints in &Theta; have no equivalent in the stock Gymnasium spaces, which is why the
leaf spaces are custom: an EV allocation is capped per stat **and** by a shared budget; `moves`,
`ability`, `gender` and `tera_type` are only legal for particular species; and poke-env identifies
Pokemon by fuzzy name matching, so a team whose species cannot be told apart produces corrupted
observations even though Showdown accepts it.

## Rejection

`reset` reports a design it cannot battle with rather than raising, because a design an optimizer
proposed is data, not a programming error. What a rejected candidate is worth is the optimizer's
decision, not the environment's:

```python
observation, info, embodiment_rejected = env.reset(seed=0, embodiment=candidate)
if embodiment_rejected.any():
    optimizer.tell(candidate, my_rejection_loss)
    print(info["embodiment_rejection_reason"])
    # pokemon_0: Alcremie-Ruby-Cream has no legal moves 'Absorb'
```

`embodiment_rejected` holds one flag per environment and is also in the info, alongside the reason.
A rejected reset returns a zero observation with a single legal action, so a masked policy still
forms a distribution over it. Stepping an environment whose design was rejected raises: it never
started a battle.

There are two separate questions about a design, and they cost very different amounts:

```python
env.embodiment_space.contains(design)   # ~45 us   is it in Theta?
env.validate_embodiment(design)         # ~236 ms  will Showdown build it?      -> bool
env.validate_embodiments(population)    # ~18 ms each, batched                  -> reasons
```

The second is not implied by the first. `contains` judges each parameter on its own, so it cannot
see that two moves are individually legal for a species but not learnable *together*: Showdown
rejects Arbok holding both Body Slam and Disable, because no single Arbok could have been on both
acquisition paths. Only Showdown settles that, so screen candidates with `validate_embodiments`
before spending episodes on them.

## The flat representation

Optimizers usually want a fixed-width vector rather than a dictionary. `flat_space` is a
`MultiDiscrete` view, and `to_index` / `from_index` convert:

```python
row = env.embodiment_space.to_index(design)        # -> (embodiment_dim,) of int64
design = env.embodiment_space.from_index(row)      # exact inverse, or raises
```

`from_index(to_index(design)) == design` always. Decoding is **strict**: a point that names no
design raises rather than being adjusted into a nearby one, so the design battled is always the
design proposed.

That matters because `flat_space` is a *box* and &Theta; is a strict subset of it. A box cannot
say that EVs share a budget, that species must be distinguishable, or that a move must come from
the pool of the species beside it. For a six-Pokemon Gen 9 design the box holds about 10<sup>180</sup>
points and &Theta; about 10<sup>59</sup> of them, so uniformly sampling the box essentially never
lands on a design. Sample &Theta; directly with `sample()` and mutate from there; rejection costs
about 50&nbsp;us and never reaches Showdown.

A vector environment takes a whole population at once:

```python
vec = pce.make_vec("random", num_envs=64)
batch = np.stack([space.to_index(space.sample()) for _ in range(64)])   # (num_envs, embodiment_dim)
observations, infos, rejected = vec.reset(embodiment=batch)             # (num_envs,) of bool
```

`reset` accepts a `(num_envs, embodiment_dim)` array or a sequence of dictionaries for one design
per environment, and a single array row or dictionary to give every environment the same design.
Omitting it keeps whatever is installed, which is what the automatic reset after a finished
episode relies on.

The vector environment is a `gymnasium.vector.VectorEnv`, so it returns what the single environment
returns: `reset` gives `(observations, infos)` and `step` gives observations, rewards,
terminations, truncations and infos. Infos are Gymnasium's vectorized form, one array per key, so a
rejection is read straight off `infos["embodiment_rejected"]`. Finished environments are reset
inside the step that ended them, which the environment declares as
`metadata["autoreset_mode"] = AutoresetMode.SAME_STEP`; the observation that ended an episode
arrives as `infos["final_obs"]`.

An environment whose design was rejected never started a battle, so stepping the vector refuses
outright rather than simulating nothing:

```
RuntimeError: environments [1] were given an embodiment that was rejected, so they have no battle
to step; reset them with designs this environment accepts
```

Stable-Baselines3 expects its own vector interface. `examples/utils.py` has an `SB3VecEnvAdapter`
that translates, so the benchmark itself depends on no training framework.

## Choosing a coordinate system

Indices are resolved against the **union** of a parameter's values, so an index names the same
move whatever species sits beside it, and an illegal pairing is reported as one:

```
union[157] = 'Energy Ball'      for every species
pokemon_0: Lilligant-Hisui has no legal ability 'Sweet Veil'
```

The alternative is to index each species' own legal list, so index 5 means *the 5th move this
species can learn*. Both reach exactly the same &Theta; &mdash; they are two coordinate systems on
one design space, not two design spaces &mdash; and they trade off against each other:

| | union (canonical) | species-relative |
| --- | --- | --- |
| meaning of an index | fixed | changes with the species |
| mutating a move index | ~91% rejected | more often legal |
| crossover | usually infeasible offspring | feasible offspring, but moves are reinterpreted |
| what a rejection means | "this species cannot learn this move" | often just "index past this species' pool" |

The environment ships the union form because a rejection then carries information, and because a
fixed coordinate system keeps results comparable across methods. Neither form is dense once
projection is off: a `MultiDiscrete` slot needs one fixed width, and legal move pools range from 1
to 232 values, so any single width leaves most points outside &Theta;.

### Converting between them

Both directions need only `legal_values`, which is public. Species-relative to union is total and
injective; union to species-relative is partial, defined exactly where the pairing is legal.

```python
def union_values(space, param):
    return space.spaces[space.pokemon_keys[0]][param].values


def to_union(space, param, species, relative_index):
    """Species-relative index -> union index. Total and injective."""
    value = space.legal_values(param, species)[relative_index]
    return union_values(space, param).index(value)


def to_species_relative(space, param, species, union_index):
    """Union index -> species-relative index. None when the pairing is illegal."""
    value = union_values(space, param)[union_index]
    legal = space.legal_values(param, species)
    return legal.index(value) if value in legal else None
```

For Garchomp in Gen 9 OU, which can learn 63 of the 632 moves in the union:

```
species-relative -> union -> species-relative is the identity: True
union -> species-relative defined for 63/632 union indices
union index 5 = 'Acrobatics' -> None            (Garchomp cannot learn it)
species-relative index 5 = 'Bulldoze' -> union 65
```

Wrap those in a `gymnasium.Wrapper` over `embodiment_space` to search in whichever coordinates
suit your designer. &Theta; is unchanged either way, so results stay comparable.

## Stepping

`step(action, embodiment=None)` takes one battle action and returns observation, reward,
termination, truncation and info, as any Gymnasium environment does. It refuses rather than
simulating something meaningless in three cases: before the first `reset`, after a `reset` whose
design was rejected, and for an action in the padded part of the action space, which is masked
illegal in every observation and so can only be reached by ignoring the mask.

## Transition-level changes

Some codesign environments allow the embodiment to change between transitions. This one does not:
the team is fixed once a battle begins, so `PokEnv.SUPPORTS_TRANSITION_EMBODIMENT` is `False` and
`step(action, embodiment=...)` warns and ignores the embodiment. Change a design at `reset`.

## Seeding

`reset(seed=...)` fixes both teams. The designer resolves every random and defaulted attribute of
the designed team from it, and the opponent's team comes from Showdown's generator, which is given
a seed drawn from the same stream. Two runs with the same seed therefore start from the same
position:

```python
observation, info, rejected = env.reset(seed=42, embodiment=design)
```

It seeds a *stream* rather than freezing a team: consecutive battles in one run still face
different opponents, and the whole sequence repeats when the seed does.

What a seed does not fix is the battle itself. Damage rolls, critical hits and speed ties are the
Showdown server's own, and are not reachable from here, so a seed reproduces the matchup rather
than the outcome.

## Servers

`make` and `make_vec` launch the Pokemon Showdown servers their environments need and shut them
down when the environment is closed. There is no option to attach to a server someone else
started: ports are claimed from the operating system so concurrent runs never collide, and a
`base_port` you force is an error if anything already holds it.

`make_vec` launches one shard per `envs_per_shard` environments (32 by default) and spreads the
workers over them round-robin.
