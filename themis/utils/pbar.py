import asyncio
import contextlib
from contextlib import asynccontextmanager

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


@asynccontextmanager
async def get_async_progress():
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=10,
    )

    async def progress_updater():
        with progress:
            while not progress.finished:
                await asyncio.sleep(0.1)
                progress.refresh()

    update_task = asyncio.create_task(progress_updater())
    try:
        yield progress
    finally:
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            contextlib.suppress(asyncio.CancelledError)


def get_progress_bar() -> Progress:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    return progress
