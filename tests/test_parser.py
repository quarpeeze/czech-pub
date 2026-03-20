import os
import sys

print("cwd:", os.getcwd())
print("sys.path[0]:", sys.path[0])

from evaluation.parser import parse_mcq_answer


VALID = ["A", "B", "C", "D"]


def check(raw_text, expected_label, expected_status):
    result = parse_mcq_answer(raw_text, VALID)
    assert result.label == expected_label, f"{raw_text!r}: expected label {expected_label}, got {result.label}"
    assert result.status == expected_status, f"{raw_text!r}: expected status {expected_status}, got {result.status}"


def test_exact_short_answers():
    check("A", "A", "ok")
    check("b", "B", "ok")
    check("(C)", "C", "ok")
    check("D.", "D", "ok")
    check("  a  ", "A", "ok")


def test_longer_answer_markers_czech():
    check("Odpověď: A", "A", "ok")
    check("Odpoved: B", "B", "ok")
    check("Odpověď je C", "C", "ok")
    check("Odpoved je D", "D", "ok")
    check("Správná odpověď je A", "A", "ok")
    check("Spravna odpoved: B", "B", "ok")


def test_longer_choice_markers():
    check("Volba C", "C", "ok")
    check("správná volba: D", "D", "ok")
    check("Možnost A", "A", "ok")
    check("Moznost: B", "B", "ok")
    check("Varianta C", "C", "ok")
    check("Option D", "D", "ok")


def test_letter_markers():
    check("Písmeno A", "A", "ok")
    check("Pismeno: B", "B", "ok")


def test_final_letter_fallback():
    check("Moje finální odpověď je C", "C", "ok")
    check("Vybral bych D", "D", "ok")


def test_empty_cases():
    check(None, None, "empty")
    check("", None, "empty")
    check("   ", None, "empty")


def test_invalid_cases():
    check("Ano", None, "invalid")
    check("Nevím", None, "invalid")
    check("Odpověď je E", None, "invalid")
    check("Volba Z", None, "invalid")
    check("Písmeno X", None, "invalid")
    check("Správná odpověď je", None, "invalid")


def test_ambiguous_cases():
    check("A nebo B", None, "ambiguous")
    check("A anebo C", None, "ambiguous")
    check("A or B", None, "ambiguous")
    check("Odpověď: A. Ale možná B.", None, "ambiguous")
    check("Volba C, případně D", None, "ambiguous")


def main():
    """
    iterate through the funcitons and print the results
    """
    test_functions = [
        test_exact_short_answers,
        test_longer_answer_markers_czech,
        test_longer_choice_markers,
        test_letter_markers,
        test_final_letter_fallback,
        test_empty_cases,
        test_invalid_cases,
        test_ambiguous_cases,
    ]

    for f in test_functions:
        try:
            f()
            print(f"{f.__name__}: ok")
        except AssertionError as e:
            print(f"{f.__name__}: not_ok; {e}")
        except Exception as e:
            print(f"{f.__name__}: not_ok")
            print(f"  unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()