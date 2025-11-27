# attendance.py
# Simple attendance utilities: mark_present(name), clear_today(), get_marked()
# Writes a daily CSV and also creates an XLS version (xls updated from CSV).
# Folder: ./attendance/
import os
import csv
from datetime import datetime
import xlwt
from threading import Lock

_attendance_folder = os.path.join(os.getcwd(), "attendance")
os.makedirs(_attendance_folder, exist_ok=True)

def _today_basename():
    return datetime.now().strftime("%Y-%m-%d")

def _csv_path():
    return os.path.join(_attendance_folder, _today_basename() + ".csv")

def _xls_path():
    return os.path.join(_attendance_folder, _today_basename() + ".xls")

_marked_today = set()
_lock = Lock()

def _ensure_csv_exists():
    path = _csv_path()
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Timestamp"])
    return path

def _csv_append(name, timestamp):
    path = _ensure_csv_exists()
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, timestamp])

def _csv_read_all():
    path = _csv_path()
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(r)
    return rows

def _write_xls_from_csv():
    rows = _csv_read_all()
    book = xlwt.Workbook()
    sheet = book.add_sheet("Attendance")
    for r_idx, row in enumerate(rows):
        for c_idx, cell in enumerate(row):
            sheet.write(r_idx, c_idx, cell)
    book.save(_xls_path())

def mark_present(name):
    """
    Mark the person present for today.
    - Name duplicates for today are ignored (won't append twice).
    - Appends to today's CSV and re-generates XLS.
    """
    if not name:
        return False
    name = str(name).strip()
    if name == "":
        return False
    with _lock:
        if name in _marked_today:
            return False
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _csv_append(name, ts)
        _marked_today.add(name)
        try:
            _write_xls_from_csv()
        except Exception:
            pass
    return True

def clear_today():
    with _lock:
        _marked_today.clear()
        path = _csv_path()
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Timestamp"])
        try:
            _write_xls_from_csv()
        except Exception:
            pass
    return True

def get_marked():
    with _lock:
        return sorted(list(_marked_today))
