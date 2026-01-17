#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

def analyze_daily_usage():
    log_file = Path("/home/david-rodriguez/customer_bot/app/logs/usage.jsonl")
    
    if not log_file.exists():
        print("❌ No usage logs yet")
        return
    
    today = datetime.now().date()
    queries = []
    
    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts.date() == today:
                    queries.append(entry)
            except:
                continue
    
    if not queries:
        print(f"📭 No queries today ({today})")
        return
    
    total = len(queries)
    avg_time = sum(q["response_time_sec"] for q in queries) / total
    times = [q["response_time_sec"] for q in queries]
    
    print(f"\n📊 Daily Report - {today}")
    print("=" * 50)
    print(f"Total Queries:      {total}")
    print(f"Avg Response Time:  {avg_time:.1f}s")
    print(f"Fastest:            {min(times):.1f}s")
    print(f"Slowest:            {max(times):.1f}s")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    analyze_daily_usage()
