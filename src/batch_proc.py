import multiprocessing
from queue import Empty
import time
import torch

class BatchProcessor:
    def __init__(self, tasks, producer_func, consumer_func, batch_len, num_producers=None, precompute_func=None):
        if num_producers is None:
            self.num_producers = multiprocessing.cpu_count()-2
        else:
            self.num_producers = num_producers
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.completion_event = multiprocessing.Event()
        self.producer_func = producer_func
        self.consumer_func = consumer_func
        self.batch_len = batch_len
        self.num_producers = num_producers
        self.kernels = None if precompute_func is None else precompute_func()

        # Load tasks into the task queue
        for task in tasks:
            self.task_queue.put(task)

    def producer(self):
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get(timeout=1)
                while self.result_queue.qsize() > 2 * self.batch_len:
                    time.sleep(1)
                result = self.producer_func(task)
                if result is not None:
                    self.result_queue.put(result)
            except Empty:
                break  # No more tasks
        self.completion_event.wait()
        print('ending')
    def consumer(self):
        batch = []
        while True:
            try:
                result = self.result_queue.get(timeout=10)
                batch.append(result)
                if len(batch) == self.batch_len:
                    if self.kernels:
                        self.consumer_func(batch, self.kernels)
                    else:
                        self.consumer_func(batch)
                    batch = []
                    torch.cuda.empty_cache()
            except Empty:
                if batch:
                    if self.kernels:
                        self.consumer_func(batch, self.kernels)
                    else:
                        self.consumer_func(batch)
                break  # Exit if no more results

    def start(self):
        # Start producer processes
        producers = [multiprocessing.Process(target=self.producer) for _ in range(self.num_producers)]
        for p in producers:
            p.start()

        # Start consumer in the main process
        self.consumer()
        torch.cuda.empty_cache()
        self.completion_event.set()
        # Wait for all producers to finish
        for p in producers:
            p.join()
