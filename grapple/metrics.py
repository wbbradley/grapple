import socket
from typing import Dict, Optional


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
        tags: Optional[Dict[str, str]],
    ) -> None:
        if tags:
            tag_str = ",".join(f"{k}:{v}" for k, v in tags.items())
            message = f"{metric}:{value}|{metric_type}|# {tag_str}"
        else:
            message = f"{metric}:{value}|{metric_type}"
        self.sock.sendto(message.encode(), (self.host, self.port))

    def close(self) -> None:
        self.sock.close()


client = StatsdClient()


def metrics_count(
    metric: str, value: float = 1, tags: Optional[Dict[str, str]] = None
) -> None:
    # Track custom metrics
    client.send_metric(metric, value, "count", tags)
