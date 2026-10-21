"""Interactive human-versus-engine play.

The engine is fully observable to a program but not to a player: Order tokens and
secret Power bids must stay hidden until the rules reveal them. This module
therefore keeps two concerns apart:

* the board renderer hides every Order token that is not the human's while
  ``state.orders_revealed`` is false;
* the network seats search with at least two determinizations, so their PIMC
  worlds resample the human's face-down information instead of reading it.

The desktop application lives in :mod:`agot.gui`; the command::

    agot play --checkpoint checkpoints/run3 --house stark

opens it. :func:`play_interactive` is the small input/output controller shared
by the GUI and the tests. PyTorch is imported only for ``opponents="net"``;
the greedy and random fallbacks work on a torch-free install.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agot.agents import MCTS, GreedyPolicy, Policy, RandomPolicy
from agot.engine.actions import Action, ActionKind
from agot.engine.enums import ORDERS_BY_KEY, OrderType, Phase, Track, UnitType
from agot.engine.game import Game, GameConfig
from agot.engine.observation import UNIT_ORDER

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]

_UNIT_ABBR = {
    UnitType.FOOTMAN: "F",
    UnitType.KNIGHT: "K",
    UnitType.SIEGE: "E",
    UnitType.SHIP: "S",
}
_ORDER_NAMES = {
    OrderType.MARCH: "March",
    OrderType.DEFEND: "Defend",
    OrderType.SUPPORT: "Support",
    OrderType.RAID: "Raid",
    OrderType.CONSOLIDATE_POWER: "Consolidate Power",
}
_TRACK_NAMES = {
    Track.IRON_THRONE: "Iron Throne",
    Track.FIEFDOMS: "Fiefdoms",
    Track.KINGS_COURT: "King's Court",
}
_HELP = """Commands:
  <number>  play the numbered action
  b         redraw the board
  ?         show this help
  u         undo your previous decision
  q         quit"""


@dataclass(frozen=True)
class PlayResult:
    """The outcome of one interactive session."""

    human_house: str
    winner: str | None
    rounds: int
    steps: int
    aborted: bool = False
    eliminated: bool = False


def resolve_checkpoint(path: str | Path) -> str:
    """Resolve a checkpoint file or a run directory.

    A directory is convenient for numbered run folders: it picks
    ``agot-best.pt`` when the run has evaluated at least once, otherwise the
    latest ``agot.pt``.
    """
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        for name in ("agot-best.pt", "agot.pt"):
            checkpoint = candidate / name
            if checkpoint.is_file():
                return str(checkpoint)
    elif candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"no checkpoint at {candidate}; pass a .pt file or a run directory"
    )


def _pretty(value: str) -> str:
    return value.replace("_", " ").title()


def _house_name(house: str | None) -> str:
    return _pretty(house) if house else "Neutral"


def _region_name(game: Game, region_id: str | None) -> str:
    if not region_id:
        return "?"
    region = game.map.regions.get(region_id)
    return region.name if region is not None else _pretty(region_id)


def order_label(key: str | None) -> str:
    if not key:
        return "?"
    spec = ORDERS_BY_KEY.get(key)
    if spec is None:
        return _pretty(key)
    label = _ORDER_NAMES.get(spec.order_type, _pretty(spec.order_type.value))
    if spec.order_type is OrderType.MARCH and spec.strength:
        label += f" {spec.strength:+d}"
    elif spec.order_type is OrderType.DEFEND:
        label += f" {spec.strength}"
    elif spec.order_type is OrderType.SUPPORT and spec.strength:
        label += f" +{spec.strength}"
    if spec.starred:
        label += "*"
    # The two physical copies of an order are functionally identical, but
    # showing the copy makes a legal-action list much easier to scan.
    if key.endswith("_a"):
        label += " (A)"
    elif key.endswith("_b"):
        label += " (B)"
    return label


def _phase_label(state) -> str:
    label = _pretty(state.phase.value)
    if state.phase is Phase.ACTION:
        label += f" ({_pretty(state.action_step.value)})"
    return label


def _unit_name(unit: str | UnitType | None) -> str:
    if unit is None:
        return "?"
    if isinstance(unit, UnitType):
        return unit.value.title()
    try:
        return UnitType(unit).value.title()
    except ValueError:
        return _pretty(str(unit))


def _unit_counts(counts: dict[UnitType, int]) -> str:
    pieces = [
        f"{_UNIT_ABBR[unit]}{counts[unit]}"
        for unit in UNIT_ORDER
        if counts.get(unit)
    ]
    return " ".join(pieces)


def _action_unit_counts(units: tuple[int, ...] | None) -> str:
    if units is None:
        return "all units"
    pieces = [
        f"{_UNIT_ABBR[unit]}{count}"
        for unit, count in zip(UNIT_ORDER, units, strict=False)
        if count
    ]
    return " ".join(pieces) or "no units"


def format_action(game: Game, action: Action) -> str:
    """A human-readable description of one legal action."""
    kind = action.kind
    region = _region_name(game, action.region)
    target = _region_name(game, action.target)

    if kind is ActionKind.PASS:
        if not action.region:
            return "Pass"
        if game.state.phase is Phase.PLANNING:
            return f"Decline to place an order in {region}"
        return f"Decline the action in {region}"
    if kind is ActionKind.PLACE_ORDER:
        return f"Place {order_label(action.order)} in {region}"
    if kind is ActionKind.RESOLVE_RAID:
        return f"Raid {target} from {region}"
    if kind is ActionKind.MARCH:
        text = (
            f"March {_action_unit_counts(action.units)} "
            f"from {region} to {target}"
        )
        return text + (" (establish control)" if action.establish else "")
    if kind is ActionKind.RESOLVE_CONSOLIDATE:
        return f"Collect Power in {region}"
    if kind is ActionKind.CONSOLIDATE_MUSTER:
        return f"Muster in {region}"
    if kind is ActionKind.PLAY_HOUSE_CARD:
        if action.card:
            return f"Play House card {_pretty(action.card)}"
        return "Play no House card"
    if kind is ActionKind.USE_BLADE:
        return "Use the Valyrian Steel Blade"
    if kind is ActionKind.RAVEN_SWAP:
        return f"Messenger Raven: change the order in {region} to {order_label(action.order)}"
    if kind is ActionKind.RAVEN_DECK_TOP:
        return "Messenger Raven: put the drawn Westeros card on top of the deck"
    if kind is ActionKind.RAVEN_DECK_BOTTOM:
        return "Messenger Raven: put the drawn Westeros card on the bottom of the deck"
    if kind is ActionKind.RETREAT:
        return f"Retreat from {region} to {target}"
    if kind is ActionKind.DISBAND:
        return f"Disband one {_unit_name(action.unit)} in {region}"
    if kind is ActionKind.CHOOSE_CASUALTY:
        return f"Destroy one {_unit_name(action.unit)} in {region}"
    if kind is ActionKind.DECLARE_SUPPORT:
        if action.side == "refuse":
            return f"Refuse to support from {region}"
        return f"Support the {action.side or '?'} from {region}"
    if kind is ActionKind.MUSTER:
        text = f"Muster one {_unit_name(action.unit)} in {region}"
        return text + (f" (to {target})" if action.target else "")
    if kind is ActionKind.UPGRADE:
        return f"Upgrade to one {_unit_name(action.unit)} in {region}"
    if kind is ActionKind.BID:
        return f"Bid {action.amount if action.amount is not None else '?'} Power"
    if kind is ActionKind.WESTEROS_CHOICE:
        if action.card and action.card != "none":
            return f"Resolve Westeros card {_pretty(action.card)}"
        return "Choose no Westeros effect"
    return str(action)


def render_board(game: Game, human_house: str | None = None) -> str:
    """Render the visible position, with other Houses' face-down orders hidden."""
    state = game.state
    human_house = human_house.lower() if human_house else None
    lines = ["=" * 88]
    active = _house_name(state.active_house) if state.active_house else "-"
    you = _house_name(human_house) if human_house else "spectator"
    lines.append(
        f"Round {state.round} | {_phase_label(state)} | "
        f"active: {active} | you: {you}"
    )
    for track in Track:
        order = state.tracks.get(track, [])
        names = " > ".join(_house_name(house) for house in order) if order else "-"
        lines.append(f"{_TRACK_NAMES[track]}: {names}")
    lines.append(f"Wildling threat: {state.wildling_threat}")

    battle = state.pending_battle
    if battle:
        attacker = _house_name(battle.get("attacker"))
        defender = _house_name(battle.get("defender"))
        battle_region = _region_name(game, battle.get("region"))
        stage = _pretty(str(battle.get("stage", "?")))
        lines.append(f"Battle in {battle_region}: {attacker} vs {defender} [{stage}]")

    lines.append("")
    lines.append(
        f"{' House':<14}{'Power':>6}{'Supply':>7}{'Castles':>8}"
        f"{'Cards':>7}{'Orders':>7}"
    )
    for house_id, house in state.houses.items():
        mark = "*" if house_id == human_house else " "
        cards_total = len(house.house_cards_available) + len(house.house_cards_used)
        cards = (
            f"{len(house.house_cards_available)}/{cards_total}"
            if cards_total
            else "0"
        )
        castles = f"{state.castle_count(house_id)}/{state.victory_castles}"
        suffix = " (eliminated)" if house.eliminated else ""
        lines.append(
            f"{mark}{_house_name(house_id):<13}{house.power:>6}"
            f"{house.supply_level:>7}{castles:>8}{cards:>7}"
            f"{len(house.orders_available):>7}{suffix}"
        )

    human = state.houses.get(human_house) if human_house else None
    if human is not None:
        hand = ", ".join(_pretty(card) for card in human.house_cards_available)
        lines.append("")
        lines.append(f"Your House cards: {hand or 'none'}")

    lines.append("")
    lines.append("Regions")
    for region_id, region in state.regions.items():
        static = game.map.regions.get(region_id)
        name = static.name if static is not None else _pretty(region_id)
        controller = _pretty(region.controller) if region.controller else "-"
        details: list[str] = []
        units = _unit_counts(region.units)
        if units:
            details.append(units)
        routed = _unit_counts(region.routed)
        if routed:
            details.append(f"routed {routed}")
        if region.neutral_force:
            details.append(f"neutral {region.neutral_force}")
        if region.garrison:
            details.append(f"garrison {region.garrison}")
        if static is not None and static.is_fortification:
            details.append("castle" if static.is_castle else "stronghold")
        if region.influence_token:
            details.append(f"influence {_pretty(region.influence_token)}")
        if region.mustering_points:
            details.append(f"MP {region.mustering_points}")
        if region.order:
            visible = state.orders_revealed or (
                human_house is not None and region.order_house == human_house
            )
            token = order_label(region.order) if visible else "?"
            owner = _pretty(region.order_house) if region.order_house else "?"
            details.append(f"order {owner} [{token}]")
        if region.owner and region.owner != region.controller:
            details.append(f"owner {_pretty(region.owner)}")
        lines.append(
            f"  {name:<29}{controller:<12}" + "  ".join(details)
        )

    lines.append("")
    lines.append("Legend: F footman, K knight, E siege engine, S ship; * = your house.")
    if not state.orders_revealed:
        lines.append("Face-down orders are shown as [?].")
    lines.append("=" * 88)
    return "\n".join(lines)


def make_policies(
    config: GameConfig,
    human_house: str,
    opponents: str,
    simulations: int,
    determinizations: int,
    checkpoint: str | Path | None,
    seed: int,
) -> dict[str, Policy]:
    houses = [house for house in config.houses if house != human_house]
    if opponents == "net":
        if not checkpoint:
            raise ValueError(
                "net opponents need a checkpoint; pass --checkpoint "
                "<run dir or .pt>"
            )
        path = resolve_checkpoint(checkpoint)
        try:
            from agot.nn.model import NeuralEvaluator, PolicyValueNet
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "net opponents need PyTorch: pip install -e \".[rl]\""
            ) from error
        evaluator = NeuralEvaluator(PolicyValueNet.load(path))
        return {
            house: MCTS(
                evaluator,
                simulations=simulations,
                determinizations=determinizations,
                temperature=0.0,
                seed=seed + index * 1009,
            )
            for index, house in enumerate(houses)
        }
    if opponents == "greedy":
        return {
            house: GreedyPolicy(seed + index)
            for index, house in enumerate(houses)
        }
    if opponents == "random":
        return {
            house: RandomPolicy(seed + index)
            for index, house in enumerate(houses)
        }
    raise ValueError(
        f"unknown opponent type {opponents!r}; choose 'net', 'greedy' or 'random'"
    )


def _print_actions(game: Game, actions: list[Action], output_fn: OutputFn) -> None:
    output_fn(f"Your move ({len(actions)} options):")
    for index, action in enumerate(actions, 1):
        output_fn(f"  {index:>2}. {format_action(game, action)}")
    output_fn("  b. board   ?. help   u. undo   q. quit")


def _prompt_action(
    game: Game,
    human_house: str,
    actions: list[Action],
    *,
    can_undo: bool,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> Action | str:
    while True:
        try:
            raw = input_fn("choice> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return "quit"

        lowered = raw.lower()
        if lowered in ("b", "board"):
            output_fn(render_board(game, human_house))
            _print_actions(game, actions, output_fn)
            continue
        if lowered in ("?", "h", "help"):
            output_fn(_HELP)
            continue
        if lowered in ("q", "quit", "exit"):
            return "quit"
        if lowered in ("u", "undo"):
            if can_undo:
                return "undo"
            output_fn("There is no earlier decision to undo.")
            continue
        if not raw:
            output_fn("Enter a number, or b/?/u/q.")
            continue
        try:
            index = int(raw)
        except ValueError:
            output_fn(
                f"Not a number: {raw!r}. Enter 1-{len(actions)} or b/?/u/q."
            )
            continue
        if 1 <= index <= len(actions):
            return actions[index - 1]
        output_fn(f"Choose a number from 1 to {len(actions)}.")


def ai_action_summary(game: Game, house: str, action: Action) -> str:
    """Describe an AI decision without leaking a face-down order, bid or card."""
    if action.kind is ActionKind.PLACE_ORDER and not game.state.orders_revealed:
        return f"{_house_name(house)} places an order."
    if action.kind in (
        ActionKind.RAVEN_SWAP,
        ActionKind.RAVEN_DECK_TOP,
        ActionKind.RAVEN_DECK_BOTTOM,
    ):
        return f"{_house_name(house)} uses the Messenger Raven."
    if action.kind is ActionKind.BID:
        return f"{_house_name(house)} bids in secret."
    if action.kind is ActionKind.PLAY_HOUSE_CARD:
        return f"{_house_name(house)} selects a House card."
    return f"{_house_name(house)}: {format_action(game, action)}"


def play_interactive(
    checkpoint: str | Path | None = None,
    *,
    human_house: str = "stark",
    opponents: str = "net",
    simulations: int = 64,
    determinizations: int = 2,
    config: GameConfig | None = None,
    seed: int = 0,
    input_fn: InputFn | None = None,
    output_fn: OutputFn | None = None,
    max_steps: int | None = None,
) -> PlayResult:
    """Play one game as ``human_house`` against AI seats.

    ``input_fn`` and ``output_fn`` are injectable for tests and other front
    ends. ``max_steps`` is a test/debug stop, not a game rule.

    The AI seats use two determinizations by default. A single determinization
    is faster, but then the search sees the human's face-down Order tokens and
    secret bids, so it is not a fair game.
    """
    if input_fn is None:
        input_fn = input
    if output_fn is None:
        output_fn = print

    human_house = human_house.lower()
    opponents = opponents.lower()
    if simulations < 1:
        raise ValueError("simulations must be at least 1")
    if determinizations < 1:
        raise ValueError("determinizations must be at least 1")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps cannot be negative")
    if config is None:
        config = GameConfig(seed=seed)
    if human_house not in config.houses:
        raise ValueError(
            f"unknown house {human_house!r}; choose from {list(config.houses)}"
        )

    policies = make_policies(
        config,
        human_house,
        opponents,
        simulations,
        determinizations,
        checkpoint,
        seed,
    )
    game = Game.new(config)

    output_fn("A Game of Thrones - interactive play")
    output_fn(
        f"You are {_house_name(human_house)}; the other houses use {opponents}."
    )
    if opponents == "net":
        output_fn(
            f"AI search: {simulations} simulations over "
            f"{determinizations} determinization(s)."
        )
    output_fn(_HELP)

    steps = 0
    aborted = False
    eliminated = False
    last_board_round: int | None = None
    undo_stack: list[tuple[Game, int]] = []

    while not game.is_terminal():
        if max_steps is not None and steps >= max_steps:
            output_fn(f"Stopped after {steps} decisions.")
            aborted = True
            break

        state = game.state
        human = state.houses.get(human_house)
        if human is not None and human.eliminated:
            eliminated = True
            break

        player = game.current_player()
        actions = game.legal_actions()
        if not actions:
            output_fn("The engine offered no legal actions; stopping.")
            aborted = True
            break

        if player == human_house:
            if last_board_round != state.round:
                output_fn(render_board(game, human_house))
                last_board_round = state.round
            output_fn("")
            output_fn(
                f"Round {state.round} | {_phase_label(state)} | your move."
            )
            _print_actions(game, actions, output_fn)
            choice = _prompt_action(
                game,
                human_house,
                actions,
                can_undo=bool(undo_stack),
                input_fn=input_fn,
                output_fn=output_fn,
            )
            if choice == "quit":
                output_fn("Game ended early.")
                aborted = True
                break
            if choice == "undo":
                if undo_stack:
                    game, steps = undo_stack.pop()
                last_board_round = None
                output_fn("Undid your last decision.")
                continue
            assert isinstance(choice, Action)
            # Store the position and step count *before* the action: undoing
            # restores the board and lets the human make a different decision.
            undo_stack.append((game.clone(), steps))
            action = choice
        else:
            if player is None:
                output_fn("The engine did not name a player; stopping.")
                aborted = True
                break
            action = policies[player].act(game)
            output_fn(f"  {ai_action_summary(game, player, action)}")

        game.step(action)
        steps += 1

    final_state = game.state
    winner = game.winner()
    rounds = final_state.round

    if eliminated:
        output_fn("")
        output_fn(render_board(game, human_house))
        output_fn(
            f"Your house was eliminated after {rounds} rounds "
            f"({steps} decisions)."
        )
    elif not aborted:
        output_fn("")
        output_fn(render_board(game, human_house))
        if winner == human_house:
            output_fn(f"You win after {rounds} rounds ({steps} decisions).")
        elif winner:
            output_fn(
                f"{_house_name(winner)} wins after {rounds} rounds "
                f"({steps} decisions)."
            )
        else:
            output_fn(
                f"Game over after {rounds} rounds with no winner "
                f"({steps} decisions)."
            )

    return PlayResult(
        human_house=human_house,
        winner=winner,
        rounds=rounds,
        steps=steps,
        aborted=aborted,
        eliminated=eliminated,
    )


__all__ = [
    "PlayResult",
    "ai_action_summary",
    "format_action",
    "make_policies",
    "order_label",
    "play_interactive",
    "render_board",
    "resolve_checkpoint",
]
