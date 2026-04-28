"""NEAT genome topology visualizer using networkx + matplotlib."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np

# Suppress networkx edge-drawing UserWarnings (API changed in networkx 3.x;
# messages start with \n so message-pattern filters fail — target by module instead)
warnings.filterwarnings("ignore", category=UserWarning, module=r"networkx")


_INPUT_COLOR = "#4A90D9"
_OUTPUT_COLOR = "#E05C5C"
_HIDDEN_COLOR = "#6DBF7E"
_DISABLED_ALPHA = 0.18


def _node_positions(genome, config) -> dict[int, tuple[float, float]]:
    """Assign (x, y) positions: inputs left, outputs right, hidden in middle."""
    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)
    # genome.nodes does not store input nodes; hidden = genome.nodes minus outputs
    hidden_keys = [k for k in genome.nodes if k not in output_keys]

    pos: dict[int, tuple[float, float]] = {}

    n_in = len(input_keys)
    for i, k in enumerate(sorted(input_keys)):
        pos[k] = (0.0, i / max(n_in - 1, 1))

    n_out = len(output_keys)
    for i, k in enumerate(sorted(output_keys)):
        pos[k] = (1.0, i / max(n_out - 1, 1))

    n_hid = len(hidden_keys)
    for i, k in enumerate(sorted(hidden_keys)):
        pos[k] = (0.5, i / max(n_hid - 1, 1))

    return pos


def draw_neat_genome(
    genome,
    config,
    output_path: str | Path,
    title: str = "",
    figsize: tuple[float, float] = (8, 5),
) -> None:
    """Draw a single NEAT genome as a layered directed graph.

    Edges: width ∝ |weight|, blue = positive, red = negative, dashed = disabled.
    Nodes: color by type (input / hidden / output), label shows bias.
    """
    input_keys = set(config.genome_config.input_keys)
    output_keys = set(config.genome_config.output_keys)

    G = nx.DiGraph()
    # input nodes live in config, not genome.nodes
    for k in input_keys:
        G.add_node(k)
    for k in genome.nodes:
        G.add_node(k)
    for (i, o), conn in genome.connections.items():
        G.add_edge(i, o, weight=conn.weight, enabled=conn.enabled)

    pos = _node_positions(genome, config)

    node_colors = []
    node_labels = {}
    for k in G.nodes:
        if k in input_keys:
            node_colors.append(_INPUT_COLOR)
            node_labels[k] = f"in{k}"
        elif k in output_keys:
            gene = genome.nodes[k]
            node_colors.append(_OUTPUT_COLOR)
            node_labels[k] = f"out{k}\nb={gene.bias:.2f}"
        else:
            gene = genome.nodes[k]
            node_colors.append(_HIDDEN_COLOR)
            node_labels[k] = f"{k}\nb={gene.bias:.2f}"

    # build edge lists by enabled/disabled
    enabled_edges = [(i, o) for (i, o), d in G.edges.items() if d["enabled"]]
    disabled_edges = [(i, o) for (i, o), d in G.edges.items() if not d["enabled"]]

    weights = [G[i][o]["weight"] for i, o in enabled_edges]
    max_w = max((abs(w) for w in weights), default=1.0)

    norm = mcolors.Normalize(vmin=-max_w, vmax=max_w)
    cmap = cm.coolwarm

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=600, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=7, font_color="white")

    # enabled edges — colored by weight
    for (i, o), w in zip(enabled_edges, weights):
        color = cmap(norm(w))
        nx.draw_networkx_edges(
            G, pos, edgelist=[(i, o)], ax=ax,
            edge_color=[color],
            width=0.8 + 2.5 * abs(w) / max(max_w, 1e-6),
            arrows=True, arrowsize=12,
            connectionstyle="arc3,rad=0.1",
            min_source_margin=18, min_target_margin=18,
        )

    # disabled edges — faint grey dashed
    if disabled_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=disabled_edges, ax=ax,
            edge_color=["#aaaaaa"],
            width=0.6, style="dashed", alpha=_DISABLED_ALPHA,
            arrows=False,
            connectionstyle="arc3,rad=0.1",
            min_source_margin=18, min_target_margin=18,
        )

    # colorbar for edge weights
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Connection weight", fontsize=8)

    n_nodes = len(genome.nodes)
    n_conn = sum(1 for c in genome.connections.values() if c.enabled)
    fitness_str = f"  fitness={genome.fitness:.1f}" if genome.fitness is not None else ""
    full_title = title or f"NEAT genome — {n_nodes} nodes, {n_conn} enabled connections{fitness_str}"
    ax.set_title(full_title, fontsize=10, pad=10)

    # legend for node types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_INPUT_COLOR, label="Input"),
        Patch(facecolor=_HIDDEN_COLOR, label="Hidden"),
        Patch(facecolor=_OUTPUT_COLOR, label="Output"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_neat_growth_sequence(
    snapshots: list[tuple[int, object]],
    config,
    output_path: str | Path,
    figsize_per: tuple[float, float] = (4.5, 4.0),
) -> None:
    """Plot NEAT topology at multiple generations side by side.

    snapshots: list of (generation, genome) tuples, e.g. [(0,g0),(50,g50),(100,g100)]
    """
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]))
    if n == 1:
        axes = [axes]

    input_keys = set(config.genome_config.input_keys)
    output_keys = set(config.genome_config.output_keys)

    for ax, (gen, genome) in zip(axes, snapshots):
        G = nx.DiGraph()
        for k in input_keys:
            G.add_node(k)
        for k in genome.nodes:
            G.add_node(k)
        for (i, o), conn in genome.connections.items():
            G.add_edge(i, o, weight=conn.weight, enabled=conn.enabled)

        pos = _node_positions(genome, config)
        node_colors = []
        for k in G.nodes:
            if k in input_keys:
                node_colors.append(_INPUT_COLOR)
            elif k in output_keys:
                node_colors.append(_OUTPUT_COLOR)
            else:
                node_colors.append(_HIDDEN_COLOR)

        enabled_edges = [(i, o) for (i, o), d in G.edges.items() if d["enabled"]]
        disabled_edges = [(i, o) for (i, o), d in G.edges.items() if not d["enabled"]]
        weights = [G[i][o]["weight"] for i, o in enabled_edges]
        max_w = max((abs(w) for w in weights), default=1.0)
        norm = mcolors.Normalize(vmin=-max_w, vmax=max_w)
        cmap = cm.coolwarm

        ax.set_axis_off()
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=350, alpha=0.92)

        for (i, o), w in zip(enabled_edges, weights):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(i, o)], ax=ax,
                edge_color=[cmap(norm(w))],
                width=0.6 + 2.0 * abs(w) / max(max_w, 1e-6),
                arrows=True, arrowsize=10,
                connectionstyle="arc3,rad=0.1",
                min_source_margin=14, min_target_margin=14,
            )
        if disabled_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=disabled_edges, ax=ax,
                edge_color=["#aaaaaa"], width=0.5,
                style="dashed", alpha=_DISABLED_ALPHA, arrows=False,
                connectionstyle="arc3,rad=0.1",
                min_source_margin=14, min_target_margin=14,
            )

        n_hid = sum(1 for k in genome.nodes if k not in input_keys and k not in output_keys)
        n_conn = sum(1 for c in genome.connections.values() if c.enabled)
        ax.set_title(f"Gen {gen}\n{n_hid} hidden, {n_conn} conn", fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_INPUT_COLOR, label="Input"),
        Patch(facecolor=_HIDDEN_COLOR, label="Hidden"),
        Patch(facecolor=_OUTPUT_COLOR, label="Output"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.85)
    fig.suptitle("NEAT Topology Growth", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_genome_onto_ax(ax, genome, config, gen: int) -> None:
    """Draw one genome frame onto an existing axis (used by the animator)."""
    ax.clear()
    ax.set_axis_off()

    input_keys = set(config.genome_config.input_keys)
    output_keys = set(config.genome_config.output_keys)

    G = nx.DiGraph()
    for k in input_keys:
        G.add_node(k)
    for k in genome.nodes:
        G.add_node(k)
    for (i, o), conn in genome.connections.items():
        G.add_edge(i, o, weight=conn.weight, enabled=conn.enabled)

    pos = _node_positions(genome, config)
    node_colors = [
        _INPUT_COLOR if k in input_keys else
        _OUTPUT_COLOR if k in output_keys else
        _HIDDEN_COLOR
        for k in G.nodes
    ]

    enabled_edges = [(i, o) for (i, o), d in G.edges.items() if d["enabled"]]
    disabled_edges = [(i, o) for (i, o), d in G.edges.items() if not d["enabled"]]
    weights = [G[i][o]["weight"] for i, o in enabled_edges]
    max_w = max((abs(w) for w in weights), default=1.0)
    norm = mcolors.Normalize(vmin=-max_w, vmax=max_w)
    cmap = cm.coolwarm

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500, alpha=0.92)

    for (i, o), w in zip(enabled_edges, weights):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(i, o)], ax=ax,
            edge_color=[cmap(norm(w))],
            width=0.8 + 2.5 * abs(w) / max(max_w, 1e-6),
            arrows=True, arrowsize=12,
            connectionstyle="arc3,rad=0.1",
            min_source_margin=16, min_target_margin=16,
        )
    if disabled_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=disabled_edges, ax=ax,
            edge_color=["#aaaaaa"], width=0.5,
            style="dashed", alpha=_DISABLED_ALPHA, arrows=False,
            connectionstyle="arc3,rad=0.1",
            min_source_margin=16, min_target_margin=16,
        )

    n_hid = sum(1 for k in genome.nodes if k not in output_keys)
    n_conn = sum(1 for c in genome.connections.values() if c.enabled)
    fitness_str = f"  fitness={genome.fitness:.1f}" if genome.fitness is not None else ""
    ax.set_title(
        f"Generation {gen} — {n_hid} hidden nodes, {n_conn} connections{fitness_str}",
        fontsize=10, pad=8,
    )


def draw_neat_animation(
    snapshots: list[tuple[int, object]],
    config,
    output_path: str | Path,
    fps: int = 4,
    figsize: tuple[float, float] = (7, 5),
) -> None:
    """Save an animated GIF of NEAT topology evolving across generations.

    snapshots: list of (generation, genome) — typically agent._genome_history
    fps:       frames per second in the output GIF
    """
    if not snapshots:
        return

    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)

    legend_elements = [
        Patch(facecolor=_INPUT_COLOR, label="Input"),
        Patch(facecolor=_HIDDEN_COLOR, label="Hidden"),
        Patch(facecolor=_OUTPUT_COLOR, label="Output"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.85)

    def update(frame_idx):
        gen, genome = snapshots[frame_idx]
        _render_genome_onto_ax(ax, genome, config, gen)

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(snapshots),
        interval=1000 // fps,
        repeat=True,
    )

    path = Path(output_path).with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer="pillow", fps=fps)
    plt.close(fig)
