"""
Wait for all EE export tasks matching a description prefix to complete.

Usage:
    python wait_for_tasks.py --prefix iran_
"""

import argparse
import time
import ee


def main():
    parser = argparse.ArgumentParser(
        description='Wait for EE export tasks to complete.')
    parser.add_argument('--prefix', required=True,
                        help='Task description prefix to match')
    parser.add_argument('--poll-interval', type=int, default=300,
                        help='Seconds between polls (default: 300)')
    parser.add_argument('--max-age-hours', type=int, default=24,
                        help='Only consider tasks created within this many hours (default: 24)')
    parser.add_argument('--project', default='ggmap-325812',
                        help='GEE cloud project ID')
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    import datetime
    cutoff_ms = int((datetime.datetime.now(datetime.timezone.utc)
                     - datetime.timedelta(hours=args.max_age_hours)).timestamp() * 1000)

    while True:
        tasks = ee.batch.Task.list()
        matching = [t for t in tasks
                    if t.config.get('description', '').startswith(args.prefix)
                    and t.status().get('creation_timestamp_ms', 0) > cutoff_ms]
        active = [t for t in matching if t.state in ('READY', 'RUNNING')]
        failed = [t for t in matching if t.state == 'FAILED']
        completed = [t for t in matching if t.state == 'COMPLETED']

        print(f"Tasks: {len(completed)} completed, {len(active)} active, "
              f"{len(failed)} failed (of {len(matching)} total)")

        if failed:
            for t in failed:
                err = t.status().get('error_message', '')
                print(f"  FAILED: {t.config.get('description')}: {err}")

        if not active:
            print("All tasks finished.")
            if failed:
                print(f"WARNING: {len(failed)} tasks failed")
            break

        time.sleep(args.poll_interval)


if __name__ == '__main__':
    main()
