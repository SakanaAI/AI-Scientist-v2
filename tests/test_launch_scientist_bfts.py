from launch_scientist_bfts import find_pdf_path_for_review


def _touch(directory, filename):
    path = directory / filename
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def test_no_pdfs_returns_none(tmp_path):
    assert find_pdf_path_for_review(str(tmp_path)) is None


def test_pdfs_present_but_none_are_reflections_returns_none(tmp_path):
    # Regression test for issue #125: this used to raise UnboundLocalError.
    _touch(tmp_path, "some_other_report.pdf")

    assert find_pdf_path_for_review(str(tmp_path)) is None


def test_single_reflection_pdf_with_no_number_falls_back_to_it(tmp_path):
    _touch(tmp_path, "paper_reflection.pdf")

    result = find_pdf_path_for_review(str(tmp_path))

    assert result == str(tmp_path / "paper_reflection.pdf")


def test_picks_highest_numbered_reflection(tmp_path):
    _touch(tmp_path, "paper_reflection_1.pdf")
    _touch(tmp_path, "paper_reflection_2.pdf")
    _touch(tmp_path, "paper_reflection_10.pdf")

    result = find_pdf_path_for_review(str(tmp_path))

    assert result == str(tmp_path / "paper_reflection_10.pdf")


def test_prefers_final_reflection_over_numbered_ones(tmp_path):
    _touch(tmp_path, "paper_reflection_1.pdf")
    _touch(tmp_path, "paper_reflection_2.pdf")
    _touch(tmp_path, "paper_reflection_final.pdf")

    result = find_pdf_path_for_review(str(tmp_path))

    assert result == str(tmp_path / "paper_reflection_final.pdf")
