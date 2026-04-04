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
def process_commitpack(output, max_samples, use_ft, context_window):
    """Download and preprocess CommitPack Python data for code world model training."""
    from .commitpack_processor import CommitPackProcessor

    processor = CommitPackProcessor()
    stats = processor.collect(
        Path("."),  # source ignored — data from HuggingFace
        Path(output),
        max_samples=max_samples,
        use_ft=use_ft,
        context_window=context_window,
    )
    click.echo(f"Processed {stats.num_transitions} transitions")
    click.echo(f"Written to {stats.output_path}")


if __name__ == "__main__":
    main()
