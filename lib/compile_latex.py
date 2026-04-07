"""
LaTeX compilation utility.
Runs pdflatex + bibtex sequence and moves the output PDF.
"""

import os
import shutil
import subprocess
import sys


def compile_latex(cwd, pdf_output=None, timeout=30):
    """Compile a LaTeX document in the given directory.

    Args:
        cwd: Directory containing template.tex
        pdf_output: Optional path to move the final PDF to. If None, stays as template.pdf.
        timeout: Timeout in seconds for each compilation step.
    """
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["bibtex", "template"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0 and command[0] == "pdflatex":
                # Log pdflatex errors but don't abort (common warnings are OK)
                for line in result.stdout.split("\n"):
                    if line.startswith("!"):
                        print(f"LaTeX error: {line}")
        except subprocess.TimeoutExpired:
            print(f"LaTeX compilation timed out after {timeout}s: {' '.join(command)}")
        except FileNotFoundError:
            print(f"Command not found: {command[0]}. Is texlive installed?")
            return False

    pdf_path = os.path.join(cwd, "template.pdf")
    if not os.path.exists(pdf_path):
        print("Failed to generate PDF.")
        return False

    if pdf_output:
        shutil.move(pdf_path, pdf_output)
        print(f"PDF saved to {pdf_output}")
    else:
        print(f"PDF generated at {pdf_path}")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compile_latex.py <directory> [output.pdf]")
        sys.exit(1)
    cwd = sys.argv[1]
    pdf_output = sys.argv[2] if len(sys.argv) > 2 else None
    success = compile_latex(cwd, pdf_output)
    sys.exit(0 if success else 1)
