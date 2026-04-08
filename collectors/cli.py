"""CLI for world model data collectors."""
import click
from pathlib import Path


@click.group()
def main():
    """World model data collectors."""
    pass


@main.command("collect-git")
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="edits.h5", help="Output HDF5 path")
@click.option("--max-commits", default=1000, help="Max commits to process")
@click.option("--context-window", default=256, help="Token context window size")
def collect_git(repo_path, output, max_commits, context_window):
    """Collect git edit sequences from a repository."""
    from .git_edit import GitEditCollector
    collector = GitEditCollector()
    stats = collector.collect(Path(repo_path), Path(output), max_commits=max_commits, context_window=context_window)
    click.echo(f"Collected {stats.num_transitions} transitions from {stats.num_sequences} commits")
    click.echo(f"Written to {stats.output_path}")


@main.command("process-commitpack")
@click.argument("output", type=click.Path(), default="commitpack_python.h5")
@click.option("--max-samples", default=None, type=int, help="Limit samples (None=all)")
@click.option("--use-ft/--use-full", default=True, help="CommitPackFT (small, default) or full CommitPack")
@click.option("--context-window", default=512, help="AST token sequence length")
@click.option("--rich-actions", is_flag=True, help="Use 15-dim AST diff actions (default: 7-dim)")
def process_commitpack(output, max_samples, use_ft, context_window, rich_actions):
    """Download and preprocess CommitPack Python data for code world model training."""
    from .commitpack_processor import CommitPackProcessor

    processor = CommitPackProcessor()
    stats = processor.collect(
        Path("."),
        Path(output),
        max_samples=max_samples,
        use_ft=use_ft,
        context_window=context_window,
        rich_actions=rich_actions,
    )
    click.echo(f"Processed {stats.num_transitions} transitions ({stats.metadata['action_dim']}-dim actions)")
    click.echo(f"Written to {stats.output_path}")


@main.command("build-trajectories")
@click.argument("output", type=click.Path(), default="trajectories.h5")
@click.option("--max-samples", default=None, type=int, help="Max CommitPack records to scan")
@click.option("--use-ft/--use-full", default=True, help="CommitPackFT (small) or full CommitPack")
@click.option("--context-window", default=512, help="AST token sequence length")
@click.option("--rich-actions", is_flag=True, help="Use 15-dim AST diff actions")
@click.option("--min-traj-len", default=4, help="Minimum transitions per trajectory")
@click.option("--max-traj-len", default=20, help="Maximum transitions per trajectory")
def build_trajectories(output, max_samples, use_ft, context_window, rich_actions,
                       min_traj_len, max_traj_len):
    """Chain CommitPack records into per-file edit trajectories."""
    from .trajectory_collector import TrajectoryCollector

    collector = TrajectoryCollector()
    stats = collector.collect(
        Path("."),
        Path(output),
        max_samples=max_samples,
        use_ft=use_ft,
        context_window=context_window,
        rich_actions=rich_actions,
        min_traj_len=min_traj_len,
        max_traj_len=max_traj_len,
    )
    meta = stats.metadata
    if "error" in meta:
        click.echo(f"Error: {meta['error']}")
    else:
        click.echo(
            f"Built {meta['num_trajectories']} trajectories "
            f"({meta['total_transitions']} transitions, "
            f"mean len {meta['mean_traj_len']:.1f}, "
            f"max len {meta['max_traj_len']})"
        )
        click.echo(f"Written to {stats.output_path}")


@main.command("chain-hdf5")
@click.argument("source_hdf5", type=click.Path(exists=True))
@click.argument("output", type=click.Path(), default="chained_trajectories.h5")
@click.option("--min-traj-len", default=4, help="Minimum transitions per chain")
@click.option("--max-traj-len", default=20, help="Maximum transitions per chain")
def chain_hdf5(source_hdf5, output, min_traj_len, max_traj_len):
    """Discover trajectory chains in an existing flat HDF5 by token matching.

    Chains rows where after_tokens[i] == before_tokens[j]. No re-download needed.
    """
    from .trajectory_collector import TrajectoryCollector

    collector = TrajectoryCollector()
    stats = collector.collect_from_hdf5(
        Path(source_hdf5),
        Path(output),
        min_traj_len=min_traj_len,
        max_traj_len=max_traj_len,
    )
    meta = stats.metadata
    if "error" in meta:
        click.echo(f"Error: {meta['error']}")
    else:
        click.echo(
            f"Found {meta['num_trajectories']} chains "
            f"({meta['total_transitions']} transitions, "
            f"mean len {meta['mean_traj_len']:.1f}, "
            f"max len {meta['max_traj_len']}, "
            f"{meta['coverage_pct']:.1f}% coverage)"
        )
        click.echo(f"Written to {stats.output_path}")


@main.command("extract-git-trajectories")
@click.argument("repos", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--output", "-o", default="git_trajectories.h5", help="Output HDF5 path")
@click.option("--context-window", default=512, help="AST token sequence length")
@click.option("--rich-actions", is_flag=True, help="Use 15-dim AST diff actions")
@click.option("--min-edits", default=4, help="Min transitions per trajectory")
@click.option("--max-commits", default=5000, help="Max commits to scan per repo")
@click.option("--max-trajs", default=500, help="Max trajectories per repo")
def extract_git_trajectories(repos, output, context_window, rich_actions,
                             min_edits, max_commits, max_trajs):
    """Extract per-file edit trajectories from local git repositories."""
    from .trajectory_collector import TrajectoryCollector

    collector = TrajectoryCollector()
    stats = collector.collect_from_git(
        [Path(r) for r in repos],
        Path(output),
        context_window=context_window,
        rich_actions=rich_actions,
        min_edits=min_edits,
        max_commits=max_commits,
        max_trajs_per_repo=max_trajs,
    )
    meta = stats.metadata
    if "error" in meta:
        click.echo(f"Error: {meta['error']}")
    else:
        click.echo(
            f"Extracted {meta['num_trajectories']} trajectories "
            f"({meta['total_transitions']} transitions) from {len(repos)} repos"
        )
        click.echo(f"Written to {stats.output_path}")


if __name__ == "__main__":
    main()
