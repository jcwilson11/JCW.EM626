ALLOWED_SPECIALS = set("!@#$%&?")

def first_error(pw: str):
    """
    Returns the first error message string if the password fails validation,
    otherwise returns None when the password is acceptable.
    """
    # Length checks
    if len(pw) < 6:
        return "non minimum length"
    if len(pw) > 12:
        return "exceeds maximum length"

    # Content checks
    has_letter = any(ch.isalpha() for ch in pw)
    has_digit = any(ch.isdigit() for ch in pw)
    has_special = any(ch in ALLOWED_SPECIALS for ch in pw)

    if not has_letter:
        return "no letters present"
    if not has_digit:
        return "no numbers present"
    if not has_special:
        return "missing required special character (!,@,#,$,%,&,?)"

    # Forbidden word check (case-insensitive)
    if "password" in pw.lower():
        return 'it contains the word "password"'

    # If all checks passed
    return None

def main():
    # Print requirements once
    print("Password requirements:")
    print(" - Minimum length: 6")
    print(" - Maximum length: 12")
    print(" - Any combination of numbers and letters (lowercase or uppercase); both must be present")
    print(" - Include at least one of these special characters: !, @, #, $, %, &, ?")
    print(' - Must NOT contain the word "password" (any case)')
    print()

    while True:
        user_input = input(
            "Enter a password with the given characteristics, or 'done' (without quotes) if you are finished:\n> "
        ).strip()

        if user_input == "done":
            print("Goodbye!")
            break

        error = first_error(user_input)
        if error is None:
            print("Password accepted!")
        else:
            print(f"Errors in your password, including {error}.")

if __name__ == "__main__":
    main()
