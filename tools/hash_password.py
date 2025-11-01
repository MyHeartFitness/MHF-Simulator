import bcrypt


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


if __name__ == "__main__":
    pwd = input("Enter a password to hash: ").strip()
    if not pwd:
        raise SystemExit("Password cannot be empty.")
    print(hash_password(pwd))
