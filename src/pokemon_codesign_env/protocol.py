from collections.abc import Awaitable, Callable


GEN4_SPORT_EFFECTS = {"Mud Sport", "Water Sport"}


def _effect_name(message: list[str]) -> str | None:
    if len(message) < 4:
        return None
    return message[3].removeprefix("move: ")


def _pokemon_slot(identifier: str) -> str:
    return identifier.partition(":")[0]


class Gen4ProtocolAdapter:
    """Translate legacy Gen 4 volatile messages into poke-env state."""

    def __init__(self):
        self._sport_sources: dict[tuple[str, str], set[str]] = {}

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


def install_gen4_protocol_adapter(player) -> None:
    """Install one stateful adapter on a poke-env player's battle callback."""
    adapter = Gen4ProtocolAdapter()
    original_callback: Callable[[list[list[str]]], Awaitable[None]] = player.ps_client._on_battle_message

    async def adapted_callback(split_messages: list[list[str]]) -> None:
        await original_callback(adapter.adapt(split_messages))

    player.ps_client._on_battle_message = adapted_callback
