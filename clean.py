import asyncio
import json
import os
from watchdog.events import FileMovedEvent, DirMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer
from config import RAW_DATASETS_FOLDER
from utils import flatten_zoopla_data, to_lowercase


class NewZooplaData(FileSystemEventHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self._queue = queue
        self._loop = loop

    def on_moved(self, event: DirMovedEvent | FileMovedEvent):
        if not event.is_directory:
            asyncio.run_coroutine_threadsafe(
                self._queue.put(event.dest_path), self._loop
            )


async def clean(queue: asyncio.Queue) -> None:
    while True:
        path = await queue.get()
        data = json.load(open(path, "rb"))
        data = to_lowercase(data)
        data = flatten_zoopla_data(data)
        print(json.dumps(data, indent=4))


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
    cleaner_task = asyncio.create_task(clean(queue))

    await asyncio.gather(watcher_task, cleaner_task)


if __name__ == "__main__":
    asyncio.run(main())
    # data = json.load(open(r"C:\Users\ADMIN\myprojects\house-price-valuator\datasets\raw-data\zoopla\zoopla-1750203709807-s1.json", "rb"))
    # data = to_lowercase(data)
    # data = flatten_zoopla_data(data)
    # print(json.dumps(data, indent=4))
