"""上游浏览器指纹和请求体共用的 IANA 时区。"""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_TIMEZONE = "Asia/Shanghai"
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def validate_timezone(value: object) -> str:
    name = str(value or "").strip() or DEFAULT_TIMEZONE
    try:
        ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"无效的 upstream_timezone：{name}，请填写 IANA 时区名称") from exc
    return name


def timezone_offset_min(name: str, *, at: datetime | None = None) -> int:
    """与 JavaScript Date.getTimezoneOffset 一致：UTC 减本地时间，随夏令时变化。"""
    zone = ZoneInfo(name)
    local = at.astimezone(zone) if at is not None else datetime.now(zone)
    return -int(local.utcoffset().total_seconds() / 60)


def browser_time(name: str, *, at: datetime | None = None) -> str:
    zone = ZoneInfo(name)
    local = at.astimezone(zone) if at is not None else datetime.now(zone)
    label = local.tzname()
    if name in {"America/Los_Angeles", "US/Pacific"}:
        # 与浏览器 en-US Intl.DateTimeFormat(timeZoneName="long") 及油猴脚本一致。
        label = "Pacific Daylight Time" if local.dst() else "Pacific Standard Time"
    # 不受服务器 LC_TIME 影响，浏览器 Date.toString 使用英文星期与月份。
    date = f"{_DAYS[local.weekday()]} {_MONTHS[local.month - 1]} {local.day:02d} {local.year:04d}"
    return date + local.strftime(" %H:%M:%S GMT%z") + f" ({label})"
