import http.cookiejar
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from .api import fetch_dataset

FILES = {
    "package": "hackathonlicence.zip",
    "borders": "t_dict_municipal.rar",
    "population": "rosstat_population.zip",
    "labor": "rosstat_labor.zip",
    "prices": "emiss_31052.xml",
    "mobility": "indeks-mobilnosti.csv",
}
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(url, path):
    if path.exists():
        print(f"exists {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    partial.rename(path)
    print(f"saved {path}")


def find_bsdtar():
    windows = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "tar.exe"
    for candidate in [shutil.which("bsdtar"), windows if windows.exists() else None, shutil.which("tar")]:
        if candidate:
            return str(candidate)
    raise RuntimeError("bsdtar is required to unpack the borders archive")


def unpack_rar(path, folder):
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run([find_bsdtar(), "-xf", str(path), "-C", str(folder)], check=True)


def unescape(text):
    return re.sub(r"\\u([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), text)


def emiss_grid(page):
    script = unescape(page[page.index("new FGrid({"):])
    script = script[:script.index("grid.init()")]
    headers = list(re.finditer(r"(\d+):\s*\{\s*title:\s*'([^']*)',\s*all:\s*\w+,\s*values:", script))
    filters = {}
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(script)
        values = re.findall(r"(\d+):\s*\{\s*title:\s*'([^']*)',\s*order:", script[header.end():end])
        filters[header.group(1)] = {"title": header.group(2), "values": [value for value, _ in values]}
    layout = {}
    for name in ["left_columns", "top_columns", "filterObjectIds"]:
        found = re.search(name + r":\s*\[([^\]]*)\]", script)
        layout[name] = re.findall(r"\d+", found.group(1))
    return filters, layout


def fetch_emiss(url, years, path):
    if path.exists():
        print(f"exists {path}")
        return
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = list(HEADERS.items())
    page = opener.open(url, timeout=300).read().decode("utf-8")
    token = re.search(r'name="token" value="(\w+)"', page).group(1)
    filters, layout = emiss_grid(page)
    params = [("id", url.rstrip("/").split("/")[-1]), ("title", "data"), ("struts.token.name", "token"), ("token", token)]
    params += [("lineObjectIds", v) for v in layout["left_columns"]]
    params += [("columnObjectIds", v) for v in layout["top_columns"]]
    params += [("filterObjectIds", v) for v in layout["filterObjectIds"]]
    for dimension, spec in filters.items():
        values = [v for v in spec["values"] if spec["title"] != "Год" or int(v) in years]
        params += [("selectedFilterIds", f"{dimension}_{v}") for v in values]
    base = urllib.parse.urlsplit(url)
    request = urllib.request.Request(
        f"{base.scheme}://{base.netloc}/indicator/downloadData.do?format=sdmx",
        data=urllib.parse.urlencode(params).encode(),
        headers={"Referer": url},
    )
    body = opener.open(request, timeout=600).read()
    if not body.lstrip().startswith(b"<?xml"):
        raise RuntimeError(f"EMISS returned no data for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    print(f"saved {path}")


def download(config, names):
    raw = config["paths"]["raw"]
    for name in names:
        if name == "prices":
            fetch_emiss(config["sources"]["prices"], config["prices"]["years"], raw / FILES["prices"])
        elif name == "mobility":
            fetch_dataset(config["sources"]["mobility"], raw / FILES["mobility"])
        else:
            fetch(config["sources"][name], raw / FILES[name])
    if "borders" in names:
        unpack_rar(raw / FILES["borders"], raw / "borders")