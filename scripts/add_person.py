"""
scripts/add_person.py — Add, list, or delete persons in the database.

Usage:
    python scripts/add_person.py --list
    python scripts/add_person.py --add "Sara" --role authorized
    python scripts/add_person.py --delete "Sara"   (GDPR erasure)
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database.db import init_db, insert_person_if_missing, list_persons, delete_person


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--add",    metavar="NAME")
    g.add_argument("--delete", metavar="NAME")
    g.add_argument("--list",   action="store_true")
    parser.add_argument("--role", choices=["authorized", "unauthorized"])
    args = parser.parse_args()

    init_db()

    if args.add:
        if not args.role:
            parser.error("--role is required with --add")
        insert_person_if_missing(args.add, args.role)
        label = args.add.replace(" ", "-").lower()
        insert_person_if_missing(label, args.role)
        print(f"Added '{args.add}' as '{args.role}'")
        print(f"Next: collect faces then retrain.")

    elif args.delete:
        confirm = input(f"Delete ALL data for '{args.delete}'? (yes/no): ")
        if confirm.strip().lower() == "yes":
            delete_person(args.delete)
            delete_person(args.delete.replace(" ", "-").lower())
        else:
            print("Aborted.")

    elif args.list:
        persons = list_persons()
        if not persons:
            print("No persons registered.")
        else:
            print(f"\n{'ID':<5} {'Name':<20} {'Status'}")
            print("─" * 35)
            for p in persons:
                print(f"{p['id']:<5} {p['name']:<20} {p['status']}")


if __name__ == "__main__":
    main()
