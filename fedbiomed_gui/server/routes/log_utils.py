import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

FilterValue = Union[str, datetime, None]


@dataclass
class LogQuery:
    page_size: int
    current_page: int
    max_total: Optional[int]
    skip: int
    filters: Dict[str, FilterValue]
    start_dt: Optional[datetime]
    end_dt: Optional[datetime]
    exhausted: bool = False


def parse_timestamp(
    value: Any, extra_formats: Tuple[str, ...] = ()
) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    s = str(value).strip()
    if not s:
        return None

    if s.isdigit():
        try:
            ts = float(s)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    try:
        iso_value = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(iso_value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    formats = (*extra_formats, "%Y-%m-%d %H:%M:%S,%f")
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            local_tz = datetime.now().astimezone().tzinfo
            return dt.replace(tzinfo=local_tz).astimezone(timezone.utc)
        except Exception:
            continue

    return None


def parse_log_query_args(
    args: Any,
    *,
    filter_names: Tuple[str, ...] = (),
    level_values: Optional[set] = None,
) -> Tuple[Optional[LogQuery], Optional[str]]:
    try:
        page_size = int(args.get("page_size", 200))
    except Exception:
        return None, "Invalid 'page_size'"

    try:
        current_page = int(args.get("current_page", 0))
    except Exception:
        return None, "Invalid 'current_page'"

    if current_page < 0:
        return None, "Invalid 'current_page'"

    max_num_arg = args.get("max_num_of_logs")
    max_total: Optional[int] = None
    if max_num_arg not in (None, ""):
        try:
            max_total = int(max_num_arg)
        except Exception:
            return None, "Invalid 'max_num_of_logs'"
        if max_total <= 0:
            max_total = None

    page_size = max(1, min(page_size, 2000))
    skip = current_page * page_size

    filters: Dict[str, FilterValue] = {
        "contains": args.get("contains"),
        "start_dt": None,
        "end_dt": None,
    }

    for name in filter_names:
        value = args.get(name)
        if value and name == "level":
            value = str(value).upper()
            if level_values is not None and value not in level_values:
                return None, f"Invalid '{name}'"
        filters[name] = value

    start_dt = None
    end_dt = None
    start_ts = args.get("start_ts")
    end_ts = args.get("end_ts")

    if start_ts:
        start_dt = parse_timestamp(start_ts)
        if start_dt is None:
            return None, "Invalid 'start_ts'"
        filters["start_dt"] = start_dt

    if end_ts:
        end_dt = parse_timestamp(end_ts)
        if end_dt is None:
            return None, "Invalid 'end_ts'"
        filters["end_dt"] = end_dt

    if start_dt and end_dt and start_dt > end_dt:
        return None, "Invalid date range: start_ts is after end_ts"

    if (start_dt or end_dt) and max_total is None:
        max_total = 5000

    exhausted = max_total is not None and skip >= max_total
    if max_total is not None and not exhausted:
        page_size = min(page_size, max_total - skip)

    return (
        LogQuery(
            page_size=page_size,
            current_page=current_page,
            max_total=max_total,
            skip=skip,
            filters=filters,
            start_dt=start_dt,
            end_dt=end_dt,
            exhausted=exhausted,
        ),
        None,
    )


def exhausted_payload(query: LogQuery) -> Dict[str, Any]:
    next_skip = query.max_total or query.skip
    return {
        "items": [],
        "next_skip": next_skip,
        "files": [],
        "page_size": query.page_size,
        "current_page": query.current_page,
        "next_page": query.current_page,
    }


def get_log_files(log_dir: str, basename: str) -> List[Dict[str, Any]]:
    try:
        entries = os.listdir(log_dir)
    except FileNotFoundError:
        return []

    files = []
    for name in entries:
        if not name.startswith(basename):
            continue

        full = os.path.join(log_dir, name)
        if not os.path.isfile(full):
            continue

        try:
            st = os.stat(full)
            files.append(
                {"name": name, "size": int(st.st_size), "mtime": int(st.st_mtime)}
            )
        except OSError:
            continue

    files.sort(key=lambda item: item.get("mtime", 0), reverse=True)
    return files


def file_date_from_name(name: str, basename: str) -> Optional[datetime]:
    prefix = basename + "."
    if not name.startswith(prefix):
        return None

    suffix = name[len(prefix) :]
    date_part = suffix.split(".", 1)[0]
    try:
        dt = datetime.strptime(date_part, "%Y-%m-%d")
        local_tz = datetime.now().astimezone().tzinfo
        return dt.replace(tzinfo=local_tz)
    except Exception:
        return None


def select_log_paths(log_dir: str, basename: str, query: LogQuery) -> List[str]:
    files = get_log_files(log_dir, basename)
    if not files:
        return []

    if query.start_dt is None and query.end_dt is None:
        current = os.path.join(log_dir, basename)
        return [current] if os.path.isfile(current) else []

    start_date = query.start_dt.astimezone().date() if query.start_dt else None
    end_date = query.end_dt.astimezone().date() if query.end_dt else None
    today_local = datetime.now().astimezone().date()

    include_current = True
    if start_date and today_local < start_date:
        include_current = False
    if end_date and today_local > end_date:
        include_current = False

    selected = []
    for f in files:
        name = f.get("name")
        if not name:
            continue

        if name == basename:
            if include_current:
                selected.append(name)
            continue

        file_dt = file_date_from_name(name, basename)
        if file_dt is None:
            selected.append(name)
            continue

        file_date = file_dt.date()
        if start_date and file_date < start_date:
            continue
        if end_date and file_date > end_date:
            continue

        selected.append(name)

    return [os.path.join(log_dir, name) for name in selected]


def tail_lines(path: str, max_lines: int, block_size: int = 8192) -> List[str]:
    if max_lines <= 0:
        return []

    try:
        with open(path, "rb") as fp:
            fp.seek(0, os.SEEK_END)
            pos = fp.tell()
            buffer = b""
            lines: List[bytes] = []

            while pos > 0 and len(lines) <= max_lines:
                read_size = block_size if pos >= block_size else pos
                pos -= read_size
                fp.seek(pos)
                chunk = fp.read(read_size)
                buffer = chunk + buffer
                parts = buffer.split(b"\n")
                buffer = parts[0]
                if len(parts) > 1:
                    lines = parts[1:] + lines

            if buffer:
                lines = [buffer] + lines

            decoded = [ln.decode("utf-8", errors="replace").rstrip() for ln in lines]
            return [ln for ln in decoded if ln][-max_lines:]
    except FileNotFoundError:
        return []


def read_log_items(
    paths: List[str],
    query: LogQuery,
    parse_lines: Callable[[List[str]], List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    window = min(50000, max(5000, query.skip + (query.page_size * 5)))
    items: List[Dict[str, Any]] = []
    scanned_files: List[str] = []

    for path in paths:
        scanned_files.append(os.path.basename(path))
        lines = tail_lines(path, window)
        items.extend(parse_lines(lines))
        if len(items) >= query.skip + (query.page_size * 10):
            break

    return items, scanned_files


def filter_sort_page(
    items: List[Dict[str, Any]],
    query: LogQuery,
    matcher: Callable[[Dict[str, Any], Dict[str, FilterValue]], bool],
) -> Tuple[List[Dict[str, Any]], int, int]:
    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    decorated = [
        (parse_timestamp(item.get("timestamp")) or min_dt, item) for item in items
    ]
    decorated.sort(key=lambda item: item[0], reverse=True)

    filtered = [item for _, item in decorated if matcher(item, query.filters)]
    page = filtered[query.skip : query.skip + query.page_size]
    next_skip = query.skip + len(page)
    next_page = query.current_page + 1 if page else query.current_page

    return page, next_skip, next_page
