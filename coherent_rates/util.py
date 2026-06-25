import pickle
import types
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import update_wrapper
from pathlib import Path
from typing import Generic, Literal, ParamSpec, TypeVar, overload

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure


@dataclass(frozen=True, kw_only=True)
class CamColor:
    """A class to hold CAM color palettes."""

    light: str
    warm: str
    base: str
    dark: str


CAM_BLUE = CamColor(
    light="#D1F9F1",
    warm="#00BDB6",
    base="#8EE8D8",
    dark="#133844",
)
CAM_CHERRY = CamColor(
    light="#F2CAD8",
    warm="#E18AAC",
    base="#CD3572",
    dark="#911449",
)
CAM_CREST = CamColor(
    light="#FFE2C8",
    warm="#FFC392",
    base="#FD8153",
    dark="#DD3025",
)
CAM_PURPLE = CamColor(
    light="#F2ECF8",
    warm="#D1B7EB",
    base="#A368DF",
    dark="#681FB1",
)
CAM_INDIGO = CamColor(
    light="#EBEDFB",
    warm="#B0B9F1",
    base="#5366E0",
    dark="#29347A",
)
CAM_GREEN = CamColor(
    light="#DFF2EA",
    warm="#AFDFCB",
    base="#4DB78C",
    dark="#13553A",
)


CAM_SLATE_1 = "#ECEEF1"


def setup_rc_params_thesis() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Utopia"],
            "text.latex.preamble": r"\usepackage{fourier}"
            "\n"
            r"\usepackage{amsmath}",
            "font.size": 11,
        },
    )


def get_thesis_fig_size() -> tuple[float, float]:
    total_textwidth_pt = 437.5
    pt_to_inch = 1 / 72.27

    # We want half width
    plot_width_in = (total_textwidth_pt / 2) * pt_to_inch

    # Height using Golden Ratio (Height = Width * 0.618)
    plot_height_in = plot_width_in * 0.85
    return plot_width_in, plot_height_in


def get_thesis_figure(
    *,
    fig_size: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    setup_rc_params_thesis()
    fig, ax = plt.subplots(
        figsize=fig_size or get_thesis_fig_size(),
        layout="constrained",
    )
    ax.set_facecolor(CAM_SLATE_1)
    fig.set_facecolor((0, 0, 0, 0))

    ax.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=9,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )

    # 3. Handle Label Sizes
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    return fig, ax


def get_paper_fig_size() -> tuple[float, float]:
    return (3.3, 2.5)


def setup_rc_params_paper() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,  # Use external LaTeX
            "pgf.rcfonts": False,  # Ignore Matplotlib's internal font settings
            "font.family": "serif",
            "font.serif": ["Charter"],  # Match your \usepackage{charter}
            "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{charter}                    % Main text font
        \usepackage{mathptmx}                   % Math font to match your preamble
        \usepackage{mathtools}                  % For complex math if needed
    """,
            "font.size": 9,  # Matches your 9pt document class
        },
    )


def get_paper_figure(
    *,
    fig_size: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    setup_rc_params_paper()
    fig, ax = plt.subplots(
        figsize=fig_size or get_paper_fig_size(),
        layout="constrained",
    )
    ax.set_facecolor(CAM_SLATE_1)
    fig.set_facecolor((0, 0, 0, 0))

    ax.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=8,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )

    # 3. Handle Label Sizes
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    return fig, ax


def get_paper_isf_figure(
    *,
    fig_size: tuple[float, float] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    setup_rc_params_paper()
    fig, (ax0, ax1) = plt.subplots(
        nrows=2,
        figsize=fig_size or get_paper_fig_size(),
        layout="constrained",
        sharex=True,
    )
    ax0.set_facecolor(CAM_SLATE_1)
    ax1.set_facecolor(CAM_SLATE_1)
    fig.set_facecolor((0, 0, 0, 0))

    ax0.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=8,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )
    ax1.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=8,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )

    # 3. Handle Label Sizes
    ax0.xaxis.label.set_fontsize(9)
    ax0.yaxis.label.set_fontsize(9)
    ax1.xaxis.label.set_fontsize(9)
    ax1.yaxis.label.set_fontsize(9)
    return fig, (ax0, ax1)


def format_axis_scientific(ax: Axis) -> None:
    formatter = ticker.ScalarFormatter(useMathText=True)
    # 2. Force scientific notation
    # (0, 0) tells it to use scientific notation for all numbers regardless of size
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.set_major_formatter(formatter)


CallType = Literal[
    "load_or_call_cached",
    "load_or_call_uncached",
    "call_uncached",
    "call_cached",
]


def _reduce_typevars(obj: object) -> tuple[type, tuple[str]]:
    # This tells pickle: "If you see a TypeVar, just treat it as its name string"
    # This prevents the 'typing.M' lookup entirely.
    return str, (str(obj),)


P = ParamSpec("P")
R = TypeVar("R")


class CachedFunction(Generic[P, R]):
    """A function wrapper which is used to cache the output."""

    def __init__(
        self,
        function: Callable[P, R],
        path: Path | Callable[P, Path | None] | None,
        *,
        default_call: CallType = "load_or_call_cached",
    ) -> None:
        self._inner = function
        self.Path = path

        self.default_call: CallType = default_call

    def _get_cache_path(self, *args: P.args, **kw: P.kwargs) -> Path | None:
        cache_path = self.Path(*args, **kw) if callable(self.Path) else self.Path  # ty:ignore[call-top-callable]
        if cache_path is None:
            return None
        return cache_path

    def call_uncached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function, without using the cache."""
        return self._inner(*args, **kw)

    def call_cached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function, and save the result to the cache."""
        obj = self.call_uncached(*args, **kw)
        cache_path = self._get_cache_path(*args, **kw)
        if cache_path is not None:
            buffers = list[pickle.PickleBuffer]()
            with cache_path.open("wb") as f:
                pickler = pickle.Pickler(
                    f,
                    pickle.HIGHEST_PROTOCOL,
                    buffer_callback=buffers.append,
                )
                pickler.dispatch_table = pickle.dispatch_table.copy()  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
                pickler.dispatch_table[TypeVar] = _reduce_typevars  # type: ignore[attr-defined]
                pickler.dispatch_table[types.GenericAlias] = _reduce_typevars  # type: ignore[attr-defined]
                pickler.dump(obj)

            np.savez(cache_path.with_suffix(".buffer.npz"), *buffers)
        return obj

    def _load_cache(self, *args: P.args, **kw: P.kwargs) -> R | None:
        """Load data from cache."""
        cache_path = self._get_cache_path(*args, **kw)
        if cache_path is None:
            return None

        try:
            buffer_data = np.load(cache_path.with_suffix(".buffer.npz"))

            def _get_buffer() -> Generator[memoryview, memoryview]:
                i = 0
                while True:
                    try:
                        yield buffer_data[f"arr_{i}"]
                    except KeyError:
                        break
                    i += 1

            buffers = _get_buffer()
        except FileNotFoundError:
            buffers = list[pickle.PickleBuffer]()

        try:
            with cache_path.open("rb") as f:
                unpickler = pickle.Unpickler(f, buffers=buffers)  # noqa: S301
                return unpickler.load()
        except FileNotFoundError:
            return None

    def load_or_call_uncached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function uncached, using the cached data if available."""
        obj = self._load_cache(*args, **kw)

        if obj is None:
            obj = self.call_uncached(*args, **kw)
        return obj

    def load_or_call_cached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function cached, using the cached data if available."""
        obj = self._load_cache(*args, **kw)

        if obj is None:
            obj = self.call_cached(*args, **kw)
        return obj

    def delete_cache(self, *args: P.args, **kw: P.kwargs) -> None:
        """Delete the cache file if it exists."""
        cache_path = self._get_cache_path(*args, **kw)
        if cache_path is None:
            return
        if cache_path.exists():
            cache_path.unlink()
        buffer_path = cache_path.with_suffix(".buffer.npz")
        if buffer_path.exists():
            buffer_path.unlink()

    def __call__(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function using the cache."""
        match self.default_call:
            case "call_cached":
                return self.call_cached(*args, **kw)
            case "call_uncached":
                return self.call_uncached(*args, **kw)
            case "load_or_call_cached":
                return self.load_or_call_cached(*args, **kw)
            case "load_or_call_uncached":
                return self.load_or_call_uncached(*args, **kw)


@overload
def cached(
    path: Path | None,
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]: ...


@overload
def cached(
    path: Callable[P, Path | None],
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]: ...


def cached(
    path: Path | Callable[P, Path | None] | None,
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]:
    """Cache the response of the function at the given path using pickle.

    Parameters
    ----------
    path : Path | Callable[P, Path]
        The file to read.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]

    """

    def _cached(f: Callable[P, R]) -> CachedFunction[P, R]:
        return update_wrapper(  # type: ignore aaa
            CachedFunction(f, path, default_call=default_call),  # ty:ignore[invalid-argument-type]
            f,
        )

    return _cached
