from collections.abc import Awaitable, Callable


GEN4_SPORT_EFFECTS = {"Mud Sport", "Water Sport"}
GEN4_INFORMATIONAL_ABILITIES = {"Persistent"}
CALLED_MOVE_SOURCES = {"Copycat", "Metronome", "Nature Power", "Round", "Sleep Talk"}


def _effect_name(message: list[str]) -> str | None:
    if len(message) < 4:
        return None
    return message[3].removeprefix("move: ")


def _pokemon_slot(identifier: str) -> str:
    return identifier.partition(":")[0]


def _source_move(message: list[str]) -> str | None:
    for value in reversed(message[4:]):
        if value.startswith(("[from] move: ", "[from]move: ")):
            return value.split(": ", 1)[-1]
    return None


class Gen4ProtocolAdapter:
    """Translate legacy Gen 4 volatile messages into poke-env state."""

    def __init__(self):
        self._sport_sources: dict[tuple[str, str], set[str]] = {}
        self._indirect_moves: dict[tuple[str, str], set[str]] = {}

    def adapt(self, split_messages: list[list[str]]) -> list[list[str]]:
        if not split_messages:
            return split_messages
        room = split_messages[0][0]
        adapted = [split_messages[0]]

        for message in split_messages[1:]:
            if len(message) < 2:
                adapted.append(message)
                continue
            event = message[1]
            effect = _effect_name(message)

            if event == "move" and len(message) > 3:
                source_move = _source_move(message)
                key = (room, _pokemon_slot(message[2]))
                indirect_moves = self._indirect_moves.setdefault(key, set())
                if source_move in indirect_moves:
                    animation = [message[0], "-anim", message[2], message[3]]
                    if len(message) > 4 and not message[4].startswith("["):
                        animation.append(message[4])
                    indirect_moves.add(message[3])
                    adapted.append(animation)
                    continue
                if source_move in CALLED_MOVE_SOURCES:
                    indirect_moves.add(message[3])
            if event == "-activate" and effect in {f"ability: {ability}" for ability in GEN4_INFORMATIONAL_ABILITIES}:
                adapted.append([message[0], "-ability", message[2], effect.removeprefix("ability: ")])
                continue
            if event == "-activate" and effect == "Beat Up":
                adapted.append([message[0], "-anim", message[2], effect, *message[4:]])
                continue
            if event == "-start" and effect in GEN4_SPORT_EFFECTS:
                sources = self._sport_sources.setdefault((room, effect), set())
                was_active = bool(sources)
                sources.add(_pokemon_slot(message[2]))
                if not was_active:
                    adapted.append([message[0], "-fieldstart", f"move: {effect}", f"[of] {message[2]}"])
                continue
            if event == "-end" and effect in GEN4_SPORT_EFFECTS:
                self._remove_sport_source(room, effect, _pokemon_slot(message[2]), message[0], adapted)
                continue
            if event in {"switch", "drag", "faint"} and len(message) > 2:
                for sport_effect in GEN4_SPORT_EFFECTS:
                    self._remove_sport_source(room, sport_effect, _pokemon_slot(message[2]), message[0], adapted)

            adapted.append(message)
            if event in {"turn", "upkeep"}:
                self._clear_indirect_moves(room)
            if event in {"init", "win", "tie", "deinit"}:
                self._clear_room(room)

        return adapted

    def _remove_sport_source(
        self, room: str, effect: str, source: str, prefix: str, adapted: list[list[str]],
    ) -> None:
        key = (room, effect)
        sources = self._sport_sources.get(key)
        if not sources or source not in sources:
            return
        sources.remove(source)
        if not sources:
            self._sport_sources.pop(key)
            adapted.append([prefix, "-fieldend", f"move: {effect}"])

    def _clear_room(self, room: str) -> None:
        for key in [key for key in self._sport_sources if key[0] == room]:
            self._sport_sources.pop(key)
        self._clear_indirect_moves(room)

    def _clear_indirect_moves(self, room: str) -> None:
        for key in [key for key in self._indirect_moves if key[0] == room]:
            self._indirect_moves.pop(key)


def install_gen4_protocol_adapter(player) -> None:
    """Install one stateful adapter on a poke-env player's battle callback."""
    adapter = Gen4ProtocolAdapter()
    original_callback: Callable[[list[list[str]]], Awaitable[None]] = player.ps_client._on_battle_message

    async def adapted_callback(split_messages: list[list[str]]) -> None:
        await original_callback(adapter.adapt(split_messages))

    player.ps_client._on_battle_message = adapted_callback
