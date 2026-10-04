import io
import urllib.request
import zipfile

from .download import HEADERS


class RemoteFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position = url, 0
        request = urllib.request.Request(url, method="HEAD", headers=HEADERS)
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def readinto(self, buffer):
        if self.position >= self.size:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={**HEADERS, "Range": f"bytes={self.position}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)


def open_zip(url):
    return zipfile.ZipFile(io.BufferedReader(RemoteFile(url), buffer_size=1 << 20))