import re

REGEX_PATTERNS = {
    re.compile(r"User User\d+ logged (in|out)\.", re.IGNORECASE): "User Action",
    re.compile(r"Backup (started|ended) at .*", re.IGNORECASE): "System Notification",
    re.compile(r"Backup completed successfully\.", re.IGNORECASE): "System Notification",
    re.compile(r"System updated to version .*", re.IGNORECASE): "System Notification",
    re.compile(r"File .* uploaded successfully by user .*", re.IGNORECASE): "System Notification",
    re.compile(r"Disk cleanup completed successfully\.", re.IGNORECASE): "System Notification",
    re.compile(r"System reboot initiated by user .*", re.IGNORECASE): "System Notification",
    re.compile(r"Account with ID .* created by .*", re.IGNORECASE): "User Action",
}


def classify_with_regex(log_message: str) -> str | None:
    """Classify predictable log messages using predefined regex rules."""
    if not isinstance(log_message, str):
        return None

    for pattern, label in REGEX_PATTERNS.items():
        if pattern.search(log_message):
            return label
    return None


if __name__ == "__main__":
    print(classify_with_regex("User User123 logged in."))
    print(classify_with_regex("Backup started at 12:00."))
    print(classify_with_regex("Backup completed successfully."))
    print(classify_with_regex("System updated to version 1.0.0."))
    print(classify_with_regex("File file1.txt uploaded successfully by user user1."))
    print(classify_with_regex("Disk cleanup completed successfully."))
    print(classify_with_regex("System reboot initiated by user user1."))
    print(classify_with_regex("Account with ID 5351 created by User634."))
    print(classify_with_regex("Hey bro, chill ya!"))
