"""Desktop game interface for playing against a saved policy.

Tkinter is used deliberately: it ships with CPython, so the game needs no web
server, browser, notebook or extra dependency. The setup screen selects the
human house, the opponent type, the checkpoint and the search budget; the game
screen shows the whole board, all tracks, every legal action and a public log.

Hidden information follows the same rules as :mod:`agot.play`: the board hides
other Houses' face-down orders, the log does not name an AI's hidden order, bid
or House card, and the AI search uses at least two determinizations by default.
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

from agot.engine.actions import Action
from agot.engine.enums import ORDERS_BY_KEY, OrderType, Phase, Track, UnitType
from agot.engine.game import DEFAULT_HOUSES, Game, GameConfig
from agot.play import (
    ai_action_summary,
    format_action,
    make_policies,
    order_label,
    resolve_checkpoint,
)

# -- palette ---------------------------------------------------------------

BG = "#111418"
PANEL = "#1b2026"
PANEL_ALT = "#252b33"
PANEL_DEEP = "#0d1014"
BORDER = "#333a44"
TEXT = "#e9e1d1"
MUTED = "#929aa4"
GOLD = "#c9a75c"
GOLD_DARK = "#7c6330"
SEA = "#142a37"
SEA_SELECTED = "#203b4a"
PORT = "#1c3542"
LAND = "#25231e"
LAND_SELECTED = "#3b3425"
RED = "#b45454"
GREEN = "#668f66"

HOUSE_INFO: dict[str, dict[str, str]] = {
    "baratheon": {
        "name": "Baratheon",
        "words": "Ours is the Fury",
        "home": "Dragonstone",
        "color": "#d0aa3f",
        "text": "#17130a",
    },
    "lannister": {
        "name": "Lannister",
        "words": "Hear Me Roar!",
        "home": "Lannisport",
        "color": "#b04444",
        "text": "#fff1dd",
    },
    "stark": {
        "name": "Stark",
        "words": "Winter is Coming",
        "home": "Winterfell",
        "color": "#9aabbd",
        "text": "#11161c",
    },
    "greyjoy": {
        "name": "Greyjoy",
        "words": "We Do Not Sow",
        "home": "Pyke",
        "color": "#4d8198",
        "text": "#f2f7fa",
    },
    "tyrell": {
        "name": "Tyrell",
        "words": "Growing Strong",
        "home": "Highgarden",
        "color": "#65955a",
        "text": "#f5faef",
    },
    "martell": {
        "name": "Martell",
        "words": "Unbowed, Unbent, Unbroken",
        "home": "Sunspear",
        "color": "#cf7a35",
        "text": "#1f1409",
    },
}

TRACK_LABELS = {
    Track.IRON_THRONE: "Iron Throne",
    Track.FIEFDOMS: "Fiefdoms",
    Track.KINGS_COURT: "King's Court",
}
TRACK_ORDER = (Track.IRON_THRONE, Track.FIEFDOMS, Track.KINGS_COURT)

#: A complete, roughly north-to-south grouping of all 58 regions. Keeping it
#: explicit makes the board readable without requiring a copyrighted map image.
AREAS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "The North",
        (
            "castle_black",
            "winterfell",
            "karhold",
            "the_stony_shore",
            "white_harbor",
            "widows_watch",
            "moat_cailin",
            "greywater_watch",
            "flints_finger",
            "bay_of_ice",
            "the_shivering_sea",
            "winterfell_port",
            "white_harbor_port",
        ),
    ),
    (
        "Iron Islands",
        ("pyke", "pyke_port", "ironmans_bay", "sunset_sea"),
    ),
    (
        "Riverlands & Vale",
        (
            "seagard",
            "riverrun",
            "the_twins",
            "the_fingers",
            "the_mountains_of_the_moon",
            "the_eyrie",
            "harrenhal",
        ),
    ),
    (
        "Westerlands & Crownlands",
        (
            "lannisport",
            "lannisport_port",
            "the_golden_sound",
            "stoney_sept",
            "searoad_marches",
            "crackclaw_point",
            "blackwater",
            "kingswood",
            "kings_landing",
            "dragonstone",
            "dragonstone_port",
            "blackwater_bay",
            "shipbreaker_bay",
            "the_narrow_sea",
        ),
    ),
    (
        "The Reach & Stormlands",
        (
            "the_reach",
            "highgarden",
            "oldtown",
            "oldtown_port",
            "the_arbor",
            "redwyne_straits",
            "three_towers",
            "dornish_marches",
            "storms_end",
            "storms_end_port",
            "west_summer_sea",
        ),
    ),
    (
        "Dorne",
        (
            "the_boneway",
            "princes_pass",
            "starfall",
            "yronwood",
            "salt_shore",
            "sunspear",
            "sunspear_port",
            "sea_of_dorne",
            "east_summer_sea",
        ),
    ),
)

_UNIT_ABBR = {
    UnitType.FOOTMAN: "F",
    UnitType.KNIGHT: "K",
    UnitType.SIEGE: "E",
    UnitType.SHIP: "S",
}
_UNIT_ORDER = (UnitType.FOOTMAN, UnitType.KNIGHT, UnitType.SIEGE, UnitType.SHIP)

TILE_W = 126
TILE_H = 62
TILE_GAP = 7
BOARD_PAD = 16


@dataclass(frozen=True)
class GameSettings:
    """Everything chosen on the setup screen."""

    house: str = "stark"
    opponents: str = "net"
    checkpoint: str | None = None
    simulations: int = 64
    determinizations: int = 2
    seed: int = 0
    max_rounds: int = 10


def house_name(house: str | None) -> str:
    if not house:
        return "-"
    return HOUSE_INFO.get(house, {"name": house.title()})["name"]


def house_color(house: str | None) -> str:
    if not house:
        return BORDER
    return HOUSE_INFO.get(house, {"color": MUTED})["color"]


def house_text_color(house: str | None) -> str:
    if not house:
        return TEXT
    return HOUSE_INFO.get(house, {"text": TEXT})["text"]


def discover_checkpoint() -> str:
    """Newest ``agot-best.pt`` under ``checkpoints/``, then newest ``agot.pt``."""
    root = Path("checkpoints")
    if not root.is_dir():
        return ""
    for pattern in ("**/agot-best.pt", "**/agot.pt"):
        candidates = [path for path in root.glob(pattern) if path.is_file()]
        if candidates:
            return str(max(candidates, key=lambda path: path.stat().st_mtime))
    return ""


def order_badge(order: str | None) -> str:
    """A compact order-token label for a board tile."""
    if not order:
        return "?"
    spec = ORDERS_BY_KEY.get(order)
    if spec is None:
        return "?"
    letter = {
        OrderType.MARCH: "M",
        OrderType.DEFEND: "D",
        OrderType.SUPPORT: "S",
        OrderType.RAID: "R",
        OrderType.CONSOLIDATE_POWER: "C",
    }[spec.order_type]
    if spec.order_type is OrderType.MARCH and spec.strength:
        letter += f"{spec.strength:+d}"
    elif spec.order_type is OrderType.DEFEND:
        letter += str(spec.strength)
    elif spec.order_type is OrderType.SUPPORT and spec.strength:
        letter += f"+{spec.strength}"
    if spec.starred:
        letter += "*"
    return letter


def unit_badge(counts: dict[UnitType, int]) -> str:
    pieces = [
        f"{_UNIT_ABBR[unit]}{counts[unit]}"
        for unit in _UNIT_ORDER
        if counts.get(unit)
    ]
    return " ".join(pieces)


def phase_label(state) -> str:
    text = state.phase.value.replace("_", " ").title()
    if state.phase is Phase.ACTION:
        text += f" - {state.action_step.value.replace('_', ' ').title()}"
    return text


def region_name(game: Game, region_id: str) -> str:
    region = game.map.regions.get(region_id)
    return region.name if region is not None else region_id.replace("_", " ").title()


def region_details(game: Game, human_house: str, region_id: str) -> str:
    """Plain-language region sheet for the right-hand inspector."""
    region = game.state.regions.get(region_id)
    if region is None:
        return "That region is not on this board."
    static = game.map.regions.get(region_id)
    lines = [static.name if static is not None else region_id]
    if static is not None:
        kind = static.terrain.value.title()
        if static.is_castle:
            kind += " - Castle"
        elif static.is_stronghold:
            kind += " - Stronghold"
        if static.home:
            kind += f" - home of {house_name(static.home)}"
        lines.append(kind)
    else:
        lines.append("Region")

    if region.controller:
        control = f"Controlled by {house_name(region.controller)}"
        if region.owner and region.owner != region.controller:
            control += f" (owner {house_name(region.owner)})"
        lines.append(control)
    elif region.neutral_force:
        lines.append(f"Unclaimed - Neutral Force {region.neutral_force}")
    else:
        lines.append("Unclaimed")

    ready = unit_badge(region.units)
    if ready:
        lines.append(f"Units: {ready}")
    routed = unit_badge(region.routed)
    if routed:
        lines.append(f"Routed: {routed}")
    if region.garrison:
        lines.append(f"Garrison: {region.garrison}")
    if region.neutral_force:
        lines.append(f"Neutral Force: {region.neutral_force}")
    if region.mustering_points:
        lines.append(f"Mustering points: {region.mustering_points}")
    if region.influence_token:
        lines.append(f"Influence token: {house_name(region.influence_token)}")

    if region.order:
        visible = game.state.orders_revealed or region.order_house == human_house
        owner = house_name(region.order_house)
        if visible:
            lines.append(f"Order: {owner} - {order_label(region.order)}")
        else:
            lines.append(f"Order: {owner} - face down")
    else:
        lines.append("Order: none")

    neighbours = game.map.neighbours_of(region_id)
    lines.append("Adjacent: " + ", ".join(region_name(game, rid) for rid in neighbours))
    return "\n".join(lines)


class HouseCard(tk.Frame):
    """Clickable house-selection card."""

    def __init__(self, master: tk.Misc, house: str, command) -> None:
        info = HOUSE_INFO[house]
        super().__init__(
            master,
            bg=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            cursor="hand2",
        )
        self._house = house
        self._command = command
        self._selected = False
        self._color = info["color"]

        strip = tk.Frame(self, bg=self._color, height=5, cursor="hand2")
        strip.pack(fill="x")
        body = tk.Frame(self, bg=PANEL, cursor="hand2")
        body.pack(fill="both", expand=True, padx=14, pady=10)

        name = tk.Label(
            body,
            text=info["name"].upper(),
            bg=PANEL,
            fg=self._color,
            font=("Georgia", 17, "bold"),
            anchor="w",
            cursor="hand2",
        )
        name.pack(fill="x")
        words = tk.Label(
            body,
            text=f'"{info["words"]}"',
            bg=PANEL,
            fg=MUTED,
            font=("Georgia", 9, "italic"),
            anchor="w",
            justify="left",
            wraplength=250,
            cursor="hand2",
        )
        words.pack(fill="x", pady=(2, 6))
        footer = tk.Frame(body, bg=PANEL, cursor="hand2")
        footer.pack(fill="x")
        home = tk.Label(
            footer,
            text=info["home"],
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 8),
            anchor="w",
            cursor="hand2",
        )
        home.pack(side="left")
        self._marker = tk.Label(
            footer,
            text="",
            bg=PANEL,
            fg=GOLD,
            font=("Segoe UI", 8, "bold"),
            anchor="e",
            cursor="hand2",
        )
        self._marker.pack(side="right")

        for widget in (self, strip, body, name, words, footer, home, self._marker):
            widget.bind("<Button-1>", self._on_click)

    def _on_click(self, _event=None) -> None:
        self._command(self._house)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.configure(
            highlightbackground=self._color if selected else BORDER,
            highlightthickness=3 if selected else 1,
        )
        self._marker.configure(text="SELECTED" if selected else "")


class SetupPanel(tk.Frame):
    """House and match setup before the first decision."""

    def __init__(
        self,
        master: tk.Misc,
        app: AgotApp,
        defaults: GameSettings,
    ) -> None:
        super().__init__(master, bg=BG)
        self.app = app
        self.selected_house = defaults.house
        self._starting = False
        self._cards: dict[str, HouseCard] = {}

        self._house_var = tk.StringVar(value=defaults.house)
        self._opponent_var = tk.StringVar(value=defaults.opponents)
        self._checkpoint_var = tk.StringVar(
            value=defaults.checkpoint or discover_checkpoint()
        )
        self._simulations_var = tk.StringVar(value=str(defaults.simulations))
        self._determinizations_var = tk.StringVar(
            value=str(defaults.determinizations)
        )
        self._seed_var = tk.StringVar(value=str(defaults.seed))
        self._rounds_var = tk.StringVar(value=str(defaults.max_rounds))

        self._build()

    # -- construction ----------------------------------------------------

    def _build(self) -> None:
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True, padx=42, pady=28)

        header = tk.Frame(outer, bg=BG)
        header.pack(fill="x")
        tk.Label(
            header,
            text="A GAME OF THRONES",
            bg=BG,
            fg=GOLD,
            font=("Georgia", 30, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            header,
            text="THE BOARD GAME  -  TAKE A SEAT AGAINST THE MACHINE",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))
        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", pady=(16, 18))

        body = tk.Frame(outer, bg=BG)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=5, uniform="setup")
        body.grid_columnconfigure(1, weight=4, uniform="setup")
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 24))
        self._build_house_choice(left)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_settings(right)

    def _build_house_choice(self, parent: tk.Frame) -> None:
        tk.Label(
            parent,
            text="CHOOSE YOUR HOUSE",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            parent,
            text="You will play this seat for the whole match. The other five are bots.",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(anchor="w", pady=(2, 12))

        grid = tk.Frame(parent, bg=BG)
        grid.pack(fill="both", expand=True)
        for column in range(2):
            grid.grid_columnconfigure(column, weight=1, uniform="house")
        for row in range(3):
            grid.grid_rowconfigure(row, weight=1, uniform="house")

        for index, house in enumerate(DEFAULT_HOUSES):
            card = HouseCard(grid, house, self._select_house)
            card.grid(
                row=index // 2,
                column=index % 2,
                sticky="nsew",
                padx=(0 if index % 2 == 0 else 6, 6 if index % 2 == 0 else 0),
                pady=6,
            )
            self._cards[house] = card
        self._cards[self.selected_house].set_selected(True)

    def _build_settings(self, parent: tk.Frame) -> None:
        panel = tk.Frame(
            parent,
            bg=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        panel.pack(fill="both", expand=True, padx=(0, 2), pady=4)
        inner = tk.Frame(panel, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=22, pady=20)
        inner.grid_columnconfigure(1, weight=1)

        tk.Label(
            inner,
            text="MATCH SETTINGS",
            bg=PANEL,
            fg=GOLD,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))

        row = 1
        tk.Label(
            inner,
            text="Opponent",
            bg=PANEL,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI", 9),
        ).grid(row=row, column=0, sticky="w", pady=5)
        opponent = ttk.Combobox(
            inner,
            textvariable=self._opponent_var,
            values=("net", "greedy", "random"),
            state="readonly",
            width=18,
        )
        opponent.grid(row=row, column=1, columnspan=2, sticky="ew", pady=5)
        opponent.bind("<<ComboboxSelected>>", self._on_opponent_change)

        row += 1
        tk.Label(
            inner,
            text="Checkpoint",
            bg=PANEL,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI", 9),
        ).grid(row=row, column=0, sticky="w", pady=5)
        self._checkpoint_entry = ttk.Entry(
            inner, textvariable=self._checkpoint_var
        )
        self._checkpoint_entry.grid(row=row, column=1, sticky="ew", pady=5)
        self._checkpoint_file = ttk.Button(
            inner, text="File...", command=self._browse_checkpoint
        )
        self._checkpoint_file.grid(row=row, column=2, sticky="e", padx=(6, 0), pady=5)

        row += 1
        tk.Label(inner, text="", bg=PANEL).grid(row=row, column=0)
        self._checkpoint_folder = ttk.Button(
            inner, text="Run folder...", command=self._browse_checkpoint_dir
        )
        self._checkpoint_folder.grid(
            row=row, column=1, columnspan=2, sticky="w", pady=(0, 6)
        )

        row += 1
        tk.Label(
            inner,
            text="Search",
            bg=PANEL,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI", 9),
        ).grid(row=row, column=0, sticky="w", pady=5)
        budget = tk.Frame(inner, bg=PANEL)
        budget.grid(row=row, column=1, columnspan=2, sticky="w", pady=5)
        self._simulations_spin = ttk.Spinbox(
            budget,
            from_=1,
            to=512,
            width=6,
            textvariable=self._simulations_var,
        )
        self._simulations_spin.pack(side="left")
        tk.Label(
            budget,
            text="simulations",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
        ).pack(side="left", padx=(5, 16))
        self._determinizations_spin = ttk.Spinbox(
            budget,
            from_=1,
            to=8,
            width=4,
            textvariable=self._determinizations_var,
        )
        self._determinizations_spin.pack(side="left")
        tk.Label(
            budget,
            text="determinizations",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
        ).pack(side="left", padx=(5, 0))

        row += 1
        tk.Label(
            inner,
            text="Seed",
            bg=PANEL,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI", 9),
        ).grid(row=row, column=0, sticky="w", pady=5)
        seed = ttk.Spinbox(
            inner,
            from_=0,
            to=2_000_000_000,
            width=12,
            textvariable=self._seed_var,
        )
        seed.grid(row=row, column=1, sticky="w", pady=5)

        row += 1
        tk.Label(
            inner,
            text="Round cap",
            bg=PANEL,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI", 9),
        ).grid(row=row, column=0, sticky="w", pady=5)
        rounds = ttk.Spinbox(
            inner,
            from_=1,
            to=10,
            width=12,
            textvariable=self._rounds_var,
        )
        rounds.grid(row=row, column=1, sticky="w", pady=5)

        row += 1
        tk.Label(
            inner,
            text=(
                "The net opponent reads the selected checkpoint. Two\n"
                "determinizations keep your face-down orders and bids hidden."
            ),
            bg=PANEL,
            fg=MUTED,
            justify="left",
            font=("Segoe UI", 8),
            wraplength=370,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 6))

        row += 1
        self._start_button = ttk.Button(
            inner,
            text="START THE MATCH",
            style="Accent.TButton",
            command=self._start,
        )
        self._start_button.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(16, 8)
        )

        row += 1
        self._status = tk.Label(
            inner,
            text="",
            bg=PANEL,
            fg=MUTED,
            justify="left",
            anchor="w",
            wraplength=370,
            font=("Segoe UI", 8),
        )
        self._status.grid(row=row, column=0, columnspan=3, sticky="ew")

        tk.Label(
            inner,
            text=(
                "Tip: a run folder contains agot-best.pt and agot.pt; the best\n"
                "checkpoint is preferred automatically."
            ),
            bg=PANEL,
            fg=MUTED,
            justify="left",
            font=("Segoe UI", 8),
            wraplength=370,
        ).grid(row=row + 1, column=0, columnspan=3, sticky="w", pady=(12, 0))

        self._on_opponent_change()

    # -- callbacks -------------------------------------------------------

    def _select_house(self, house: str) -> None:
        self.selected_house = house
        self._house_var.set(house)
        for key, card in self._cards.items():
            card.set_selected(key == house)

    def _on_opponent_change(self, _event=None) -> None:
        net = self._opponent_var.get() == "net"
        state = "normal" if net else "disabled"
        self._checkpoint_entry.configure(state=state)
        self._checkpoint_file.configure(state=state)
        self._checkpoint_folder.configure(state=state)
        if net and not self._checkpoint_var.get().strip():
            self._checkpoint_var.set(discover_checkpoint())

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Select a checkpoint",
            filetypes=(("PyTorch checkpoint", "*.pt"), ("All files", "*.*")),
        )
        if path:
            self._checkpoint_var.set(path)

    def _browse_checkpoint_dir(self) -> None:
        path = filedialog.askdirectory(parent=self, title="Select a run folder")
        if path:
            self._checkpoint_var.set(path)

    def _parse_int(self, variable: tk.StringVar, name: str, minimum: int) -> int:
        try:
            value = int(variable.get())
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return value

    def _start(self) -> None:
        if self._starting:
            return
        try:
            simulations = self._parse_int(self._simulations_var, "simulations", 1)
            determinizations = self._parse_int(
                self._determinizations_var, "determinizations", 1
            )
            seed = self._parse_int(self._seed_var, "seed", 0)
            rounds = self._parse_int(self._rounds_var, "round cap", 1)
        except ValueError as error:
            self.set_status(str(error), error=True)
            return

        opponents = self._opponent_var.get()
        checkpoint = self._checkpoint_var.get().strip() or None
        if opponents == "net":
            if checkpoint is None:
                self.set_status(
                    "Choose agot-best.pt (or a run folder) for the net opponent.",
                    error=True,
                )
                return
            try:
                checkpoint = resolve_checkpoint(checkpoint)
            except OSError as error:
                self.set_status(str(error), error=True)
                return

        settings = GameSettings(
            house=self.selected_house,
            opponents=opponents,
            checkpoint=checkpoint,
            simulations=simulations,
            determinizations=determinizations,
            seed=seed,
            max_rounds=rounds,
        )
        self._starting = True
        self.set_status("Preparing the table...")
        self._start_button.configure(state="disabled")
        self.update_idletasks()
        started = self.app.start_game(settings)
        self._starting = False
        if not started and self.winfo_exists():
            self._start_button.configure(state="normal")

    def set_status(self, message: str, *, error: bool = False) -> None:
        if not self.winfo_exists():
            return
        self._status.configure(text=message, fg=RED if error else MUTED)
        self.update_idletasks()


class AgotApp:
    """The application: setup screen plus one live game screen."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        defaults: GameSettings | None = None,
    ) -> None:
        self.root = root
        self.defaults = defaults or GameSettings()
        self.setup: SetupPanel | None = None
        self.game_frame: tk.Frame | None = None
        self.game: Game | None = None
        self.policies: dict[str, object] = {}
        self.settings: GameSettings | None = None
        self.human_house = self.defaults.house
        self.steps = 0
        self.undo_stack: list[tuple[Game, int]] = []
        self.selected_region: str | None = None
        self.actions: list[Action] = []
        self.busy = False
        self.finished = False
        self.generation = 0
        self.ai_after: str | None = None
        self.board_resize_after: str | None = None

        self._configure_window()
        self._build_styles()
        self.show_setup(self.defaults)

    def _configure_window(self) -> None:
        self.root.title("A Game of Thrones - The Board Game")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(screen_w, min(1380, max(980, screen_w - 60)))
        height = min(screen_h, min(860, max(660, screen_h - 80)))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(min(980, width), min(660, height))
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        default = tkfont.nametofont("TkDefaultFont")
        default.configure(family="Segoe UI", size=9)
        style.configure(
            ".",
            background=BG,
            foreground=TEXT,
            fieldbackground=PANEL_ALT,
            bordercolor=BORDER,
            darkcolor=PANEL,
            lightcolor=PANEL,
        )
        style.configure(
            "TButton",
            background=PANEL_ALT,
            foreground=TEXT,
            borderwidth=0,
            padding=(8, 6),
            font=("Segoe UI", 9),
        )
        style.map(
            "TButton",
            background=[("active", "#343c47"), ("disabled", "#171b20")],
            foreground=[("disabled", "#5f6670")],
        )
        style.configure(
            "Accent.TButton",
            background=GOLD,
            foreground="#15120b",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 8),
        )
        style.map("Accent.TButton", background=[("active", "#e0bd78")])
        style.configure(
            "Treeview",
            background=PANEL,
            fieldbackground=PANEL,
            foreground=TEXT,
            rowheight=24,
            borderwidth=0,
        )
        style.map(
            "Treeview",
            background=[("selected", "#3b4653")],
            foreground=[("selected", GOLD)],
        )
        style.configure(
            "Treeview.Heading",
            background=PANEL_ALT,
            foreground=MUTED,
            font=("Segoe UI", 8, "bold"),
            borderwidth=0,
        )
        style.map("Treeview.Heading", background=[("active", PANEL_ALT)])
        style.configure(
            "TCombobox",
            fieldbackground=PANEL_ALT,
            background=PANEL_ALT,
            foreground=TEXT,
            arrowcolor=GOLD,
            padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL_ALT)],
            foreground=[("readonly", TEXT)],
            selectbackground=[("readonly", PANEL_ALT)],
        )
        style.configure(
            "TEntry",
            fieldbackground=PANEL_ALT,
            foreground=TEXT,
            insertcolor=TEXT,
            padding=4,
        )
        style.configure(
            "TSpinbox",
            fieldbackground=PANEL_ALT,
            foreground=TEXT,
            arrowcolor=GOLD,
            padding=3,
        )
        style.configure(
            "Vertical.TScrollbar",
            background=PANEL_ALT,
            troughcolor=PANEL_DEEP,
            arrowcolor=MUTED,
            borderwidth=0,
        )

    # -- screens ---------------------------------------------------------

    def show_setup(self, defaults: GameSettings | None = None) -> None:
        self.generation += 1
        self._cancel_ai()
        if self.game_frame is not None:
            self.game_frame.destroy()
            self.game_frame = None
        if self.setup is not None:
            self.setup.destroy()
            self.setup = None
        self.game = None
        self.policies = {}
        self.actions = []
        self.undo_stack = []
        self.selected_region = None
        self.finished = False
        self.busy = False
        self.setup = SetupPanel(self.root, self, defaults or self.defaults)
        self.setup.pack(fill="both", expand=True)
        self.root.title("A Game of Thrones - The Board Game")

    def start_game(self, settings: GameSettings) -> bool:
        if self.setup is not None:
            self.setup.set_status("Preparing the table...")
        self.root.update_idletasks()
        try:
            config = GameConfig(seed=settings.seed, max_rounds=settings.max_rounds)
            policies = make_policies(
                config,
                settings.house,
                settings.opponents,
                settings.simulations,
                settings.determinizations,
                settings.checkpoint,
                settings.seed,
            )
            game = Game.new(config)
        except (OSError, RuntimeError, ValueError, ImportError) as error:
            if self.setup is not None:
                self.setup.set_status(f"Could not start: {error}", error=True)
            return False

        self.settings = settings
        self.human_house = settings.house
        self.policies = policies
        self.game = game
        self.steps = 0
        self.undo_stack = []
        self.finished = False
        self.busy = False
        self.selected_region = self._human_home()
        self.defaults = settings

        if self.setup is not None:
            self.setup.destroy()
            self.setup = None
        self._build_game_frame()
        self._append_log(
            f"You take the {house_name(self.human_house)} seat. "
            f"Opponents: {settings.opponents}.",
            "system",
        )
        self.root.title(
            f"A Game of Thrones - {house_name(self.human_house)}"
        )
        self._refresh_all()
        self._schedule_turn()
        return True

    def _human_home(self) -> str | None:
        if self.game is None:
            return None
        for region_id, region in self.game.map.regions.items():
            if region.home == self.human_house:
                return region_id
        return None

    # -- game screen construction ---------------------------------------

    def _build_game_frame(self) -> None:
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(fill="both", expand=True)
        self.game_frame = frame
        self._build_topbar(frame)
        # Keep the chronicle at the bottom and give the board the rest; packing
        # the log after an expanding body makes it disappear on short screens.
        self._build_log(frame)

        body = tk.Frame(frame, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        body.grid_columnconfigure(0, minsize=248)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, minsize=382)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._build_left(left)

        board = tk.Frame(body, bg=BG)
        board.grid(row=0, column=1, sticky="nsew")
        self._build_board(board)

        right = tk.Frame(body, bg=BG, width=382)
        right.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        right.grid_propagate(False)
        self._build_right(right)

    def _build_topbar(self, parent: tk.Frame) -> None:
        top = tk.Frame(
            parent,
            bg=PANEL,
            height=58,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        top.pack(fill="x", padx=10, pady=(8, 6))
        top.pack_propagate(False)
        top.grid_columnconfigure(1, weight=1)

        title = tk.Frame(top, bg=PANEL)
        title.grid(row=0, column=0, sticky="w", padx=14)
        tk.Label(
            title,
            text="A GAME OF THRONES",
            bg=PANEL,
            fg=GOLD,
            font=("Georgia", 15, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            title,
            text="THE BOARD GAME",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 7, "bold"),
            anchor="w",
        ).pack(anchor="w")

        center = tk.Frame(top, bg=PANEL)
        center.grid(row=0, column=1, sticky="w", padx=18)
        self._round_label = tk.Label(
            center,
            text="ROUND 1",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 12, "bold"),
        )
        self._round_label.pack(side="left")
        self._phase_label = tk.Label(
            center,
            text="PLANNING",
            bg=PANEL,
            fg=GOLD,
            font=("Segoe UI", 9, "bold"),
        )
        self._phase_label.pack(side="left", padx=(12, 0))
        self._active_label = tk.Label(
            center,
            text="ACTIVE: -",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
        )
        self._active_label.pack(side="left", padx=(14, 0))

        right = tk.Frame(top, bg=PANEL)
        right.grid(row=0, column=2, sticky="e", padx=12)
        self._human_chip = tk.Label(
            right,
            text="YOU: STARK",
            bg=house_color("stark"),
            fg=house_text_color("stark"),
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=4,
        )
        self._human_chip.pack(side="left", padx=(0, 10))
        self._opponent_label = tk.Label(
            right,
            text="BOT: NET",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
        )
        self._opponent_label.pack(side="left", padx=(0, 12))
        ttk.Button(
            right, text="New game", command=self._new_game
        ).pack(side="left")

    def _build_left(self, parent: tk.Frame) -> None:
        tk.Label(
            parent,
            text="THE REALM",
            bg=BG,
            fg=GOLD,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(2, 6))
        self._houses_body = tk.Frame(parent, bg=BG)
        self._houses_body.pack(fill="x")

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=(12, 10))
        self._tracks_body = tk.Frame(parent, bg=BG)
        self._tracks_body.pack(fill="x")
        self._wildling_label = tk.Label(
            parent,
            text="Wildling threat: 0",
            bg=BG,
            fg=RED,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        )
        self._wildling_label.pack(fill="x", pady=(10, 0))

    def _build_board(self, parent: tk.Frame) -> None:
        head = tk.Frame(parent, bg=BG)
        head.pack(fill="x", pady=(2, 5))
        tk.Label(
            head,
            text="THE BOARD",
            bg=BG,
            fg=GOLD,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(side="left")
        tk.Label(
            head,
            text=(
                "F/K/E/S units  C castle  H stronghold  "
                "G garrison  N neutral  [?] hidden order"
            ),
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 7),
            anchor="e",
        ).pack(side="right")

        wrapper = tk.Frame(parent, bg=BG)
        wrapper.pack(fill="both", expand=True)
        self._board_canvas = tk.Canvas(
            wrapper,
            bg=PANEL_DEEP,
            highlightthickness=0,
            bd=0,
        )
        scrollbar = ttk.Scrollbar(
            wrapper,
            orient="vertical",
            command=self._board_canvas.yview,
        )
        self._board_canvas.configure(yscrollcommand=scrollbar.set)
        self._board_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._board_canvas.bind("<Configure>", self._on_board_resize)
        self._board_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._board_canvas.bind(
            "<Button-4>", lambda _event: self._board_canvas.yview_scroll(-3, "units")
        )
        self._board_canvas.bind(
            "<Button-5>", lambda _event: self._board_canvas.yview_scroll(3, "units")
        )

    def _build_right(self, parent: tk.Frame) -> None:
        status = tk.Frame(
            parent,
            bg=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        status.pack(fill="x", pady=(2, 8))
        self._turn_label = tk.Label(
            status,
            text="YOUR MOVE",
            bg=PANEL,
            fg=GOLD,
            font=("Georgia", 14, "bold"),
            anchor="w",
        )
        self._turn_label.pack(fill="x", padx=12, pady=(9, 1))
        self._turn_sub = tk.Label(
            status,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
            anchor="w",
            justify="left",
            wraplength=330,
        )
        self._turn_sub.pack(fill="x", padx=12, pady=(0, 9))

        selected = tk.Frame(parent, bg=BG)
        selected.pack(fill="x", pady=(0, 6))
        tk.Label(
            selected,
            text="SELECTED REGION",
            bg=BG,
            fg=GOLD,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill="x")
        self._detail_text = tk.Text(
            selected,
            height=8,
            width=38,
            wrap="word",
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            padx=8,
            pady=6,
            font=("Segoe UI", 8),
        )
        self._detail_text.pack(fill="x", pady=(4, 0))
        self._detail_text.configure(state="disabled")

        actions = tk.Frame(parent, bg=BG)
        action_head = tk.Frame(actions, bg=BG)
        action_head.pack(fill="x", pady=(2, 4))
        tk.Label(
            action_head,
            text="ACTIONS",
            bg=BG,
            fg=GOLD,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(side="left")
        self._action_hint = tk.Label(
            action_head,
            text="",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 7),
        )
        self._action_hint.pack(side="right")

        self._selected_action = tk.Label(
            actions,
            text="",
            bg=BG,
            fg=MUTED,
            justify="left",
            anchor="w",
            wraplength=350,
            font=("Segoe UI", 8),
        )
        self._selected_action.pack(fill="x", pady=(0, 4))

        tree_frame = tk.Frame(actions, bg=PANEL)
        tree_frame.pack(fill="both", expand=True)
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        self._action_tree = ttk.Treeview(
            tree_frame,
            columns=("n", "action"),
            show="headings",
            selectmode="browse",
            height=9,
        )
        self._action_tree.heading("n", text="#")
        self._action_tree.heading("action", text="Choose an action")
        self._action_tree.column(
            "n", width=34, minwidth=30, anchor="center", stretch=False
        )
        self._action_tree.column("action", width=300, minwidth=220, anchor="w")
        action_scroll = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self._action_tree.yview
        )
        self._action_tree.configure(yscrollcommand=action_scroll.set)
        self._action_tree.grid(row=0, column=0, sticky="nsew")
        action_scroll.grid(row=0, column=1, sticky="ns")
        self._action_tree.bind("<Double-1>", self._on_action_double_click)
        self._action_tree.bind("<Return>", lambda _event: self._play_selected())
        self._action_tree.bind(
            "<<TreeviewSelect>>", lambda _event: self._update_action_selection()
        )
        self._action_tree.bind("<Motion>", self._on_action_hover)

        buttons = tk.Frame(parent, bg=BG)
        self._play_button = ttk.Button(
            buttons,
            text="Play selected action",
            style="Accent.TButton",
            command=self._play_selected,
        )
        self._play_button.pack(side="left", fill="x", expand=True)
        self._undo_button = ttk.Button(
            buttons, text="Undo", command=self._undo
        )
        self._undo_button.pack(side="left", padx=(6, 0))

        hand = tk.Frame(parent, bg=PANEL)
        tk.Label(
            hand,
            text="YOUR HOUSE CARDS",
            bg=PANEL,
            fg=GOLD,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(7, 2))
        self._hand_label = tk.Label(
            hand,
            text="",
            bg=PANEL,
            fg=TEXT,
            justify="left",
            anchor="w",
            wraplength=330,
            font=("Segoe UI", 8),
        )
        self._hand_label.pack(fill="x", padx=10, pady=(0, 8))

        # Pack from the bottom up: the hand and buttons must stay reachable on
        # short displays, and the action tree takes whatever height is left.
        hand.pack(side="bottom", fill="x")
        buttons.pack(side="bottom", fill="x", pady=(8, 8))
        actions.pack(fill="both", expand=True, pady=(2, 0))

    def _build_log(self, parent: tk.Frame) -> None:
        frame = tk.Frame(
            parent,
            bg=PANEL,
            height=110,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        frame.pack_propagate(False)
        header = tk.Frame(frame, bg=PANEL)
        header.pack(fill="x", padx=10, pady=(5, 0))
        tk.Label(
            header,
            text="CHRONICLE",
            bg=PANEL,
            fg=GOLD,
            font=("Segoe UI", 7, "bold"),
            anchor="w",
        ).pack(side="left")
        self._log = tk.Text(
            frame,
            height=5,
            wrap="word",
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            padx=6,
            pady=2,
            font=("Segoe UI", 8),
        )
        log_scroll = ttk.Scrollbar(frame, orient="vertical", command=self._log.yview)
        self._log.configure(yscrollcommand=log_scroll.set)
        self._log.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=(0, 4))
        log_scroll.pack(side="right", fill="y", pady=(0, 4))
        self._log.configure(state="disabled")
        self._log.tag_configure("system", foreground=MUTED)
        self._log.tag_configure("human", foreground=GOLD)
        self._log.tag_configure("ai", foreground=TEXT)
        self._log.tag_configure("over", foreground="#e4c986", font=("Segoe UI", 8, "bold"))
        self._log.tag_configure("error", foreground=RED)

    # -- refresh ---------------------------------------------------------

    def _refresh_all(self) -> None:
        if self.game is None or self.game_frame is None:
            return
        self._refresh_topbar()
        self._refresh_houses()
        self._refresh_tracks()
        self._refresh_board()
        self._refresh_detail()
        self._refresh_hand()
        self._refresh_turn()

    def _refresh_topbar(self) -> None:
        if self.game is None:
            return
        state = self.game.state
        self._round_label.configure(
            text=f"ROUND {state.round} / {state.max_rounds}"
        )
        self._phase_label.configure(text=phase_label(state).upper())
        active = state.active_house
        self._active_label.configure(
            text=f"ACTIVE: {house_name(active).upper() if active else '-'}",
            fg=house_color(active) if active else MUTED,
        )
        self._human_chip.configure(
            text=f"YOU: {house_name(self.human_house).upper()}",
            bg=house_color(self.human_house),
            fg=house_text_color(self.human_house),
        )
        if self.settings is not None:
            if self.settings.opponents == "net":
                checkpoint = Path(self.settings.checkpoint or "").name
                bot = f"BOT: NET - {checkpoint}"
            else:
                bot = f"BOT: {self.settings.opponents.upper()}"
            self._opponent_label.configure(text=bot)

    def _refresh_houses(self) -> None:
        if self.game is None:
            return
        for child in self._houses_body.winfo_children():
            child.destroy()
        state = self.game.state
        headers = ("House", "P", "S", "C", "Cards", "Ord")
        for column, text in enumerate(headers):
            label = tk.Label(
                self._houses_body,
                text=text,
                bg=BG,
                fg=MUTED,
                font=("Segoe UI", 7, "bold"),
                anchor="w" if column == 0 else "e",
            )
            label.grid(row=0, column=column, sticky="ew", padx=(0, 3), pady=(0, 3))
        self._houses_body.grid_columnconfigure(0, weight=1)

        for row, house_id in enumerate(DEFAULT_HOUSES, start=1):
            house = state.houses.get(house_id)
            if house is None:
                continue
            is_human = house_id == self.human_house
            background = PANEL_ALT if is_human else BG
            color = house_color(house_id)
            if house.eliminated:
                color = "#5a5f66"
            card_total = len(house.house_cards_available) + len(house.house_cards_used)
            values = (
                f"{house.power}",
                f"{house.supply_level}",
                f"{state.castle_count(house_id)}",
                f"{len(house.house_cards_available)}/{card_total}",
                f"{len(house.orders_available)}",
            )
            name_cell = tk.Frame(self._houses_body, bg=background)
            name_cell.grid(row=row, column=0, sticky="ew", padx=(0, 3), pady=1)
            tk.Frame(name_cell, bg=color, width=4, height=16).pack(
                side="left", padx=(0, 5)
            )
            name = ("* " if is_human else "") + house_name(house_id)
            if house.eliminated:
                name += " (out)"
            tk.Label(
                name_cell,
                text=name,
                bg=background,
                fg=color if not house.eliminated else MUTED,
                font=("Segoe UI", 8, "bold" if is_human else "normal"),
                anchor="w",
            ).pack(side="left", fill="x", expand=True)
            for column, text in enumerate(values, start=1):
                tk.Label(
                    self._houses_body,
                    text=text,
                    bg=background,
                    fg=TEXT if not house.eliminated else MUTED,
                    font=("Consolas", 8),
                    anchor="e",
                ).grid(
                    row=row,
                    column=column,
                    sticky="ew",
                    padx=(0, 3),
                    pady=1,
                )
        tk.Label(
            self._houses_body,
            text="P power  S supply  C castles  Ord order tokens left",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 7),
            anchor="w",
        ).grid(row=len(DEFAULT_HOUSES) + 1, column=0, columnspan=6, sticky="w", pady=(4, 0))

    def _refresh_tracks(self) -> None:
        if self.game is None:
            return
        for child in self._tracks_body.winfo_children():
            child.destroy()
        state = self.game.state
        tk.Label(
            self._tracks_body,
            text="INFLUENCE TRACKS",
            bg=BG,
            fg=GOLD,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 4))
        for track in TRACK_ORDER:
            names = state.tracks.get(track, [])
            tk.Label(
                self._tracks_body,
                text=TRACK_LABELS[track],
                bg=BG,
                fg=MUTED,
                font=("Segoe UI", 7, "bold"),
                anchor="w",
            ).pack(fill="x")
            tk.Label(
                self._tracks_body,
                text=" > ".join(house_name(house) for house in names) if names else "-",
                bg=BG,
                fg=TEXT,
                font=("Segoe UI", 8),
                justify="left",
                anchor="w",
                wraplength=230,
            ).pack(fill="x", pady=(0, 4))

        raven = state.raven_holder()
        blade = state.blade_holder()
        tk.Label(
            self._tracks_body,
            text=f"Raven: {house_name(raven)}    Blade: {house_name(blade)}",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 7),
            anchor="w",
            wraplength=230,
        ).pack(fill="x")
        self._wildling_label.configure(
            text=f"Wildling threat: {state.wildling_threat}"
        )

    def _on_board_resize(self, _event=None) -> None:
        if self.board_resize_after is not None:
            self.root.after_cancel(self.board_resize_after)
        self.board_resize_after = self.root.after(80, self._refresh_board)

    def _refresh_board(self) -> None:
        self.board_resize_after = None
        if self.game is None or not hasattr(self, "_board_canvas"):
            return
        canvas = self._board_canvas
        yview = canvas.yview()
        canvas.delete("all")
        width = canvas.winfo_width()
        if width <= 1:
            width = 620
        cols = max(1, (width - 2 * BOARD_PAD) // (TILE_W + TILE_GAP))
        state = self.game.state
        placed: set[str] = set()
        y = 12
        groups = list(AREAS)
        known = {region_id for _title, regions in groups for region_id in regions}
        missing = [
            region_id
            for region_id in self.game.map.regions
            if region_id not in known
        ]
        if missing:
            groups.append(("Other regions", tuple(missing)))

        for title, region_ids in groups:
            region_ids = [rid for rid in region_ids if rid in state.regions]
            if not region_ids:
                continue
            rows = math.ceil(len(region_ids) / cols)
            area_height = 30 + rows * (TILE_H + TILE_GAP) + 8
            canvas.create_rectangle(
                6,
                y,
                width - 6,
                y + area_height,
                fill=PANEL,
                outline="#2b3139",
                width=1,
            )
            canvas.create_text(
                BOARD_PAD,
                y + 9,
                text=title.upper(),
                fill=GOLD,
                font=("Segoe UI", 8, "bold"),
                anchor="nw",
            )
            canvas.create_text(
                width - BOARD_PAD,
                y + 9,
                text=f"{len(region_ids)} regions",
                fill=MUTED,
                font=("Segoe UI", 7),
                anchor="ne",
            )
            for index, region_id in enumerate(region_ids):
                row, column = divmod(index, cols)
                x = BOARD_PAD + column * (TILE_W + TILE_GAP)
                tile_y = y + 27 + row * (TILE_H + TILE_GAP)
                self._draw_region(canvas, x, tile_y, region_id)
                placed.add(region_id)
            y += area_height + 10

        canvas.configure(scrollregion=(0, 0, width, y + 6))
        if yview:
            canvas.yview_moveto(yview[0])

    def _draw_region(self, canvas: tk.Canvas, x: int, y: int, region_id: str) -> None:
        assert self.game is not None
        state = self.game.state
        region = state.regions[region_id]
        static = self.game.map.regions.get(region_id)
        selected = region_id == self.selected_region
        if static is not None and static.is_sea:
            fill = SEA_SELECTED if selected else SEA
        elif static is not None and static.is_port:
            fill = SEA_SELECTED if selected else PORT
        else:
            fill = LAND_SELECTED if selected else LAND
        border = house_color(region.controller) if region.controller else BORDER
        if selected:
            border = GOLD
        tag = f"region:{region_id}"
        canvas.create_rectangle(
            x,
            y,
            x + TILE_W,
            y + TILE_H,
            fill=fill,
            outline=border,
            width=3 if selected else 2 if region.controller else 1,
            tags=(tag,),
        )
        if region.controller:
            canvas.create_rectangle(
                x,
                y,
                x + 5,
                y + TILE_H,
                fill=house_color(region.controller),
                outline="",
                tags=(tag,),
            )

        name = static.name if static is not None else region_id.replace("_", " ").title()
        canvas.create_text(
            x + 9,
            y + 6,
            text=name,
            fill=TEXT,
            font=("Segoe UI", 7, "bold"),
            anchor="nw",
            width=TILE_W - 50,
            tags=(tag,),
        )

        if region.order:
            visible = state.orders_revealed or region.order_house == self.human_house
            badge = order_badge(region.order) if visible else "?"
            badge_fill = GOLD_DARK if visible else "#4a3a22"
            badge_text = "#17130a" if visible else GOLD
            canvas.create_rectangle(
                x + TILE_W - 43,
                y + 5,
                x + TILE_W - 5,
                y + 21,
                fill=badge_fill,
                outline="",
                tags=(tag,),
            )
            canvas.create_text(
                x + TILE_W - 24,
                y + 13,
                text=badge,
                fill=badge_text,
                font=("Segoe UI", 7, "bold"),
                tags=(tag,),
            )

        if region.controller:
            controller_text = house_name(region.controller)
            controller_fill = house_color(region.controller)
        elif region.neutral_force:
            controller_text = "Neutral"
            controller_fill = RED
        else:
            controller_text = "Unclaimed"
            controller_fill = MUTED
        canvas.create_text(
            x + 9,
            y + 30,
            text=controller_text,
            fill=controller_fill,
            font=("Segoe UI", 7),
            anchor="nw",
            tags=(tag,),
        )

        units = unit_badge(region.units)
        routed = unit_badge(region.routed)
        details = []
        if units:
            details.append(units)
        if routed:
            details.append(f"r {routed}")
        if region.neutral_force:
            details.append(f"N{region.neutral_force}")
        if region.garrison:
            details.append(f"G{region.garrison}")
        canvas.create_text(
            x + 9,
            y + TILE_H - 10,
            text="  ".join(details) if details else "-",
            fill=TEXT,
            font=("Consolas", 7),
            anchor="w",
            tags=(tag,),
        )
        if static is not None and static.is_fortification:
            canvas.create_text(
                x + TILE_W - 10,
                y + TILE_H - 9,
                text="C" if static.is_castle else "H",
                fill=GOLD_DARK,
                font=("Segoe UI", 8, "bold"),
                anchor="e",
                tags=(tag,),
            )
        canvas.tag_bind(
            tag,
            "<Button-1>",
            lambda _event, rid=region_id: self._select_region(rid),
        )
        canvas.tag_bind(
            tag,
            "<Enter>",
            lambda _event: canvas.configure(cursor="hand2"),
        )
        canvas.tag_bind(
            tag,
            "<Leave>",
            lambda _event: canvas.configure(cursor=""),
        )

    def _on_mousewheel(self, event) -> None:
        if event.delta:
            self._board_canvas.yview_scroll(
                -1 if event.delta > 0 else 1, "units"
            )

    def _select_region(self, region_id: str) -> None:
        self.selected_region = region_id
        self._refresh_detail()
        self._refresh_board()
        if self.actions and not self.busy and not self.finished:
            for index, action in enumerate(self.actions):
                if action.region == region_id or action.target == region_id:
                    self._action_tree.selection_set(str(index))
                    self._action_tree.see(str(index))
                    break

    def _refresh_detail(self) -> None:
        if not hasattr(self, "_detail_text"):
            return
        self._detail_text.configure(state="normal")
        self._detail_text.delete("1.0", "end")
        if self.game is None or self.selected_region is None:
            self._detail_text.insert(
                "1.0", "Click any region on the board to inspect it."
            )
        else:
            self._detail_text.insert(
                "1.0",
                region_details(self.game, self.human_house, self.selected_region),
            )
        self._detail_text.configure(state="disabled")

    def _refresh_hand(self) -> None:
        if self.game is None:
            return
        human = self.game.state.houses.get(self.human_house)
        if human is None or not human.house_cards_available:
            text = "No House cards left."
        else:
            text = ", ".join(
                card.replace("_", " ").title()
                for card in human.house_cards_available
            )
        self._hand_label.configure(text=text)

    def _refresh_turn(self) -> None:
        if self.game is None:
            return
        state = self.game.state
        human = state.houses.get(self.human_house)
        terminal = self.game.is_terminal()
        if human is not None and human.eliminated:
            self._set_turn_text("YOUR HOUSE HAS FALLEN", "You can start a new match.")
            self._clear_actions()
            self._set_play_enabled(False)
            return
        if terminal or self.finished:
            winner = self.game.winner()
            text = "GAME OVER"
            sub = (
                f"{house_name(winner)} wins."
                if winner
                else "No house has won this game."
            )
            self._set_turn_text(text, sub)
            self._clear_actions()
            self._set_play_enabled(False)
            return

        player = self.game.current_player()
        context = self._context_text()
        if player == self.human_house:
            self.busy = False
            try:
                actions = self.game.legal_actions()
            except RuntimeError:
                actions = []
            self.actions = actions
            self._fill_actions(actions)
            subtitle = (
                f"{phase_label(state)} - choose one of the "
                f"{len(actions)} legal actions."
            )
            if context:
                subtitle += f"\n{context}"
            self._set_turn_text("YOUR MOVE", subtitle)
            self._set_play_enabled(bool(actions))
        else:
            self.busy = True
            self.actions = []
            self._clear_actions()
            subtitle = f"{phase_label(state)} - the bot is choosing a move."
            if context:
                subtitle += f"\n{context}"
            self._set_turn_text(
                f"{house_name(player).upper()} IS THINKING",
                subtitle,
            )
            self._set_play_enabled(False)
        self._undo_button.configure(
            state="normal" if self.undo_stack and not self.busy else "disabled"
        )

    def _context_text(self) -> str:
        if self.game is None:
            return ""
        state = self.game.state
        parts: list[str] = []
        battle = state.pending_battle
        if battle:
            region = region_name(self.game, str(battle.get("region", "")))
            attacker = house_name(battle.get("attacker"))
            defender_value = battle.get("defender")
            defender = house_name(defender_value) if defender_value else "Neutral"
            stage = str(battle.get("stage", "")).replace("_", " ").title()
            parts.append(f"Battle in {region}: {attacker} vs {defender} [{stage}].")
        attack = state.wildling_attack
        if attack:
            parts.append(
                f"Wildling attack in progress (strength {attack.get('strength', '?')})."
            )
        pending = state.pending_westeros
        if pending:
            cards = ", ".join(card.replace("_", " ").title() for card in pending)
            parts.append(f"Westeros: {cards}.")
        if state.westeros_choice:
            parts.append("A holders' choice Westeros card is being resolved.")
        return " ".join(parts)

    def _fill_actions(self, actions: list[Action]) -> None:
        self._action_tree.delete(*self._action_tree.get_children())
        for index, action in enumerate(actions):
            self._action_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(index + 1, format_action(self.game, action)),
            )
        if actions:
            target = 0
            if self.selected_region:
                for index, action in enumerate(actions):
                    if (
                        action.region == self.selected_region
                        or action.target == self.selected_region
                    ):
                        target = index
                        break
            self._action_tree.selection_set(str(target))
            self._action_tree.see(str(target))
            self._update_action_selection()
        else:
            self._selected_action.configure(text="")
        self._action_hint.configure(text=f"{len(actions)} legal")

    def _update_action_selection(self) -> None:
        selection = self._action_tree.selection()
        if not selection or self.game is None:
            self._selected_action.configure(text="")
            return
        index = int(selection[0])
        if 0 <= index < len(self.actions):
            text = format_action(self.game, self.actions[index])
            self._selected_action.configure(text=f"Selected: {text}")

    def _on_action_hover(self, event) -> None:
        row = self._action_tree.identify_row(event.y)
        if not row or self.game is None:
            self._update_action_selection()
            return
        index = int(row)
        if 0 <= index < len(self.actions):
            text = format_action(self.game, self.actions[index])
            self._selected_action.configure(text=f"Hover: {text}")

    def _clear_actions(self) -> None:
        self._action_tree.delete(*self._action_tree.get_children())
        self._action_hint.configure(text="")
        if hasattr(self, "_selected_action"):
            self._selected_action.configure(text="")
        self.actions = []

    def _set_play_enabled(self, enabled: bool) -> None:
        self._play_button.configure(state="normal" if enabled else "disabled")

    def _set_turn_text(self, title: str, subtitle: str) -> None:
        self._turn_label.configure(text=title)
        self._turn_sub.configure(text=subtitle)

    def _append_log(self, message: str, tag: str = "system") -> None:
        if not hasattr(self, "_log"):
            return
        self._log.configure(state="normal")
        self._log.insert("end", message + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    # -- turn loop -------------------------------------------------------

    def _schedule_turn(self) -> None:
        if self.game is None or self.game_frame is None or self.finished:
            return
        if not self.game_frame.winfo_exists():
            return
        if self.game.is_terminal():
            self._finish_game(eliminated=False)
            return
        state = self.game.state
        human = state.houses.get(self.human_house)
        if human is not None and human.eliminated:
            self._finish_game(eliminated=True)
            return
        player = self.game.current_player()
        if player == self.human_house:
            self.busy = False
            self._refresh_turn()
            return
        self.busy = True
        self._refresh_turn()
        generation = self.generation
        self.ai_after = self.root.after(
            25, lambda: self._ai_step(generation)
        )

    def _ai_step(self, generation: int) -> None:
        self.ai_after = None
        if generation != self.generation or self.game is None or self.finished:
            return
        if self.game.is_terminal():
            self._finish_game(eliminated=False)
            return
        player = self.game.current_player()
        if player == self.human_house:
            self._schedule_turn()
            return
        policy = self.policies.get(player)
        if policy is None:
            self._fail(f"No policy was built for {house_name(player)}.")
            return
        try:
            action = policy.act(self.game)
        except Exception as error:  # keep the window alive on a policy failure
            self._fail(f"The bot could not move: {error}")
            return
        self._append_log(ai_action_summary(self.game, player, action), "ai")
        try:
            self.game.step(action)
        except Exception as error:
            self._fail(f"The engine rejected the bot move: {error}")
            return
        self.steps += 1
        self._refresh_all()
        self._schedule_turn()

    def _play_selected(self) -> None:
        if (
            self.busy
            or self.finished
            or self.game is None
            or self.game.current_player() != self.human_house
        ):
            return
        selection = self._action_tree.selection()
        if not selection:
            self._set_turn_text("YOUR MOVE", "Select an action in the list first.")
            return
        index = int(selection[0])
        if not 0 <= index < len(self.actions):
            return
        action = self.actions[index]
        self.undo_stack.append((self.game.clone(), self.steps))
        self._append_log(f"You: {format_action(self.game, action)}", "human")
        try:
            self.game.step(action)
        except Exception as error:
            self.undo_stack.pop()
            self._fail(f"The engine rejected that action: {error}")
            return
        self.steps += 1
        self.busy = True
        self._refresh_all()
        self._schedule_turn()

    def _on_action_double_click(self, event) -> None:
        row = self._action_tree.identify_row(event.y)
        if row:
            self._action_tree.selection_set(row)
            self._play_selected()

    def _undo(self) -> None:
        if self.busy or not self.undo_stack or self.finished:
            return
        self._cancel_ai()
        self.generation += 1
        self.game, self.steps = self.undo_stack.pop()
        self._append_log("You stepped back to your previous decision.", "system")
        self.busy = False
        self._refresh_all()
        self._schedule_turn()

    def _new_game(self) -> None:
        current = self.settings or self.defaults
        self.show_setup(current)

    def _fail(self, message: str) -> None:
        self._append_log(message, "error")
        self._set_turn_text("SOMETHING WENT WRONG", message)
        self.finished = True
        self.busy = False
        self._set_play_enabled(False)
        messagebox.showerror("A Game of Thrones", message, parent=self.root)

    def _finish_game(self, *, eliminated: bool) -> None:
        if self.finished:
            return
        self.finished = True
        self.busy = False
        self._clear_actions()
        self._set_play_enabled(False)
        self._refresh_all()
        winner = self.game.winner() if self.game is not None else None
        if eliminated:
            title = "YOUR HOUSE HAS FALLEN"
            message = (
                f"{house_name(self.human_house)} has been eliminated. "
                "The match is over for you."
            )
        elif winner == self.human_house:
            title = "YOU WIN"
            message = f"{house_name(self.human_house)} takes the Iron Throne."
        elif winner:
            title = "GAME OVER"
            message = f"{house_name(winner)} wins the game."
        else:
            title = "GAME OVER"
            message = "The round cap was reached with no winner."
        self._set_turn_text(title, message)
        self._append_log(message, "over")
        generation = self.generation
        self.root.after(
            120,
            lambda: self._game_over_dialog(generation, message),
        )

    def _game_over_dialog(self, generation: int, message: str) -> None:
        if generation != self.generation:
            return
        messagebox.showinfo("A Game of Thrones", message, parent=self.root)

    def _cancel_ai(self) -> None:
        if self.ai_after is not None:
            try:
                self.root.after_cancel(self.ai_after)
            except tk.TclError:
                pass
            self.ai_after = None

    def _on_close(self) -> None:
        self.generation += 1
        self._cancel_ai()
        self.root.destroy()


def launch(
    *,
    checkpoint: str | None = None,
    house: str = "stark",
    opponents: str = "net",
    simulations: int = 64,
    determinizations: int = 2,
    seed: int = 0,
    max_rounds: int = 10,
) -> None:
    """Open the desktop application. The setup screen is always shown first."""
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(
            "could not open a desktop window; Tkinter needs a graphical display"
        ) from error
    defaults = GameSettings(
        house=house,
        opponents=opponents,
        checkpoint=checkpoint,
        simulations=simulations,
        determinizations=determinizations,
        seed=seed,
        max_rounds=max_rounds,
    )
    AgotApp(root, defaults=defaults)
    root.mainloop()


__all__ = [
    "AREAS",
    "AgotApp",
    "GameSettings",
    "discover_checkpoint",
    "launch",
    "order_badge",
    "region_details",
    "unit_badge",
]
