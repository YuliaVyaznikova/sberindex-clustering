import base64
import csv
import json
import time
import urllib.request
import uuid

URL = "https://sberindex.ru/api/sowa"


def decode(value, kind=None):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict) and value.get("type") == "object":
        result = {}
        for item in value["value"]:
            if item.get("type") == "longstring":
                return "".join(decode(item["value"], "longstring"))
            result[item["key"]] = decode(item["value"], item.get("type"))
        return result
    text = str(value)
    if not text.startswith("__"):
        return value
    tag, _, raw = text[2:].partition("__")
    tag = kind if kind not in (None, "longstring") else tag
    if tag == "null":
        return None
    if tag == "number":
        return float(raw)
    if tag == "boolean":
        return raw == "true"
    if tag == "string":
        return base64.b64decode(raw).decode("utf-8")
    return raw


def call(route):
    body = {"SOWA": {"method": "GET", "route": base64.b64encode(route.encode()).decode(), "data": {"type": "object", "value": []}}}
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0", "RqUID": uuid.uuid4().hex}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = json.loads(response.read().decode("utf-8"))
    return decode(raw["SOWA"]["data"])


def fetch_dataset(dataset_id, path, limit=10000):
    if path.exists():
        print(f"exists {path}")
        return
    rows, offset = [], 0
    while True:
        page = call(f"/dataset/v1/{dataset_id}?limit={limit}&offset={offset}")
        rows.extend(page["data"])
        offset += limit
        if offset >= int(page["pagination"]["total_records"]) or not page["data"]:
            break
        time.sleep(0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(page["fields"])
        writer.writerows(rows)
    print(f"saved {path}")