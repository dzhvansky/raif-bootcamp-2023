import typing

import requests


DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 8080


class HTTPClient:
    """Access models served with http-based servers."""

    host: str
    port: typing.Optional[int]

    def __init__(self, host: str = DEFAULT_HOST, port: typing.Optional[int] = DEFAULT_PORT):
        self.host = host
        self.port = port

    @property
    def base_url(self):
        prefix = "http://" if not self.host.startswith("http://") and not self.host.startswith("https://") else ""
        if self.port:
            return f"{prefix}{self.host}:{self.port}"
        return f"{prefix}{self.host}"

    def __call__(self, name: str, files: dict[str, bytes], return_raw: bool = False):  # pylint: disable=R1710
        # headers = {
        #     'content-type': 'application/json',
        # }
        ret = requests.post(f"{self.base_url}/{name}/", files=files, allow_redirects=True)
        if ret.status_code == 200:  # pylint: disable=R1705
            if return_raw:
                return ret.content
            return ret.json()
        elif ret.status_code == 400:
            raise Exception(ret.json()["error"])
        else:
            ret.raise_for_status()
        return None
