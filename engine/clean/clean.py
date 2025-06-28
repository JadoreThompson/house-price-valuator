import asyncio
import json
import os
from typing import Literal
import warnings

from sqlalchemy import insert
from watchdog.events import FileMovedEvent, DirMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from db_models import ResidentialHomes
from engine.config import CLEANED_ZOOPLA_FOLDER, RAW_DATASETS_FOLDER
from utils.db import get_db_session
from .utils import flatten_zoopla_data, parse_zoopla_data, strip_values


class NewZooplaData(FileSystemEventHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self._queue = queue
        self._loop = loop

    def on_moved(self, event: DirMovedEvent | FileMovedEvent):
        if not event.is_directory:
            asyncio.run_coroutine_threadsafe(
                self._queue.put(event.dest_path), self._loop
            )


async def persist_data(
    path: str, data: dict, source: Literal["zoopla"] = "zoopla"
) -> None:
    try:
        async with get_db_session() as sess:
            incode, outcode = data["postcode"].split(" ")
            await sess.execute(
                insert(ResidentialHomes).values(
                    **data,
                    incode=incode,
                    outcode=outcode,
                    town=data["city"],
                    source=source,
                )
            )
            await sess.commit()
    except Exception as e:
        msg = f"Error persisting data from {path} to DB: {e}"
        warnings.warn(msg)

    json.dump(
        data,
        open(os.path.join(CLEANED_ZOOPLA_FOLDER, os.path.basename(path)), "w"),
    )


def clean_zoopla(data: dict) -> dict:
    data = flatten_zoopla_data(data)
    parsed_data = parse_zoopla_data(data)
    return strip_values(parsed_data)


async def clean_queue(queue: asyncio.Queue) -> None:
    while True:
        path = await queue.get()

        i = 0
        while True:
            try:
                data = json.load(open(path, "rb"))
                break
            except PermissionError:
                await asyncio.sleep(2**i)

            cleaned_data = clean_zoopla(data)
            await persist_data(path, cleaned_data)


async def watch_folder(
    dirname: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop
) -> None:
    event_handler = NewZooplaData(queue, loop)
    observer = Observer()
    observer.schedule(event_handler, path=dirname, recursive=False)
    observer.start()

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        observer.stop()
        observer.join()


async def main() -> None:
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    dirname = os.path.join(RAW_DATASETS_FOLDER, "zoopla")

    watcher_task = asyncio.create_task(watch_folder(dirname, queue, loop))
    cleaner_task = asyncio.create_task(clean_queue(queue))

    await asyncio.gather(watcher_task, cleaner_task)


def _temp():
    folder = os.path.join(RAW_DATASETS_FOLDER, "zoopla")
    fnams = os.listdir(folder)

    for fname in fnams:
        clean_path = os.path.join(CLEANED_ZOOPLA_FOLDER, fname)

        json.dump(
            clean_zoopla(json.load(open(os.path.join(folder, fname), "rb"))),
            open(clean_path, "w"),
        )
        print("cleaned ", fname)


if __name__ == "__main__":
    asyncio.run(main())
    # _temp()
