# Copyright 2024, William Bradley, All rights reserved.
import socket
import time


class StatsdClient:
    def __init__(self, host: str = "localhost", port: int = 8125) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.host = host
        self.port = port

    def send_metric(
        self,
        metric: str,
        value: float,
        metric_type: str,
    ) -> None:
        message = f"{metric}:{value}|{metric_type}"
        self.sock.sendto(message.encode(), (self.host, self.port))

    def close(self) -> None:
        self.sock.close()


client = StatsdClient()


def metrics_count(metric: str, value: float = 1) -> None:
    client.send_metric(metric, value, "c")


def metrics_gauge(metric: str, value: float) -> None:
    client.send_metric(metric, value, "g")


class metrics_timer:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> None:
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        client.send_metric(self.name, int(elapsed_time * 1000), "ms")
