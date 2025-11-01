from streamlit_authenticator import Hasher

if __name__ == "__main__":
    pwd = input("Enter a password to hash: ").strip()
    print(Hasher([pwd]).generate()[0])
