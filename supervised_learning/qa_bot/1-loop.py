#!/usr/bin/env python3
"""1. Create the loop"""


def main():
    """fdfdfd"""
    termination_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        user_input = input("Q: ").strip()
        if user_input.lower() in termination_words:
            print("A: Goodbye")
            break
        print("A:")


if __name__ == "__main__":
    main()
