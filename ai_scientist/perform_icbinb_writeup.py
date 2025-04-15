import argparse
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import traceback
import unicodedata
import tempfile
from dotenv import load_dotenv
load_dotenv()
from ai_scientist.llm import (
    get_response_from_llm,
    extract_json_between_markers,
    create_client,
    AVAILABLE_LLMS,
)


from ai_scientist.tools.semantic_scholar import search_for_papers

from ai_scientist.perform_vlm_review import (
    generate_vlm_img_review,
    perform_imgs_cap_ref_review,
    perform_imgs_cap_ref_review_selection,
    detect_duplicate_figures,
)
from ai_scientist.vlm import create_client as create_vlm_client


def remove_accents_and_clean(s):
    # Normalize to separate accents
    nfkd_form = unicodedata.normalize("NFKD", s)
    # Remove non-ASCII characters
    ascii_str = nfkd_form.encode("ASCII", "ignore").decode("ascii")
    # Remove anything but letters, digits, underscores, colons, dashes, @, {, }, and commas
    ascii_str = re.sub(r"[^a-zA-Z0-9:_@\{\},-]+", "", ascii_str)
    # Convert to lowercase
    ascii_str = ascii_str.lower()
    return ascii_str


def compile_latex(cwd, pdf_file, timeout=30):
    print("GENERATING LATEX")

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
            print("Standard Output:\n", result.stdout)
            print("Standard Error:\n", result.stderr)
        except subprocess.TimeoutExpired:
            print(
                f"EXCEPTION in compile_latex: LaTeX timed out after {timeout} seconds."
            )
            print(traceback.format_exc())
        except subprocess.CalledProcessError:
            print(
                f"EXCEPTION in compile_latex: Error running command {' '.join(command)}"
            )
            print(traceback.format_exc())

    print("FINISHED GENERATING LATEX")

    try:
        shutil.move(osp.join(cwd, "template.pdf"), pdf_file)
    except FileNotFoundError:
        print("Failed to rename PDF.")
        print("EXCEPTION in compile_latex while moving PDF:")
        print(traceback.format_exc())


def is_header_or_footer(line):
    """
    Returns True if the line is likely a header or footer.
    Filters out:
      - Lines that are too short (< 4 characters after stripping).
      - Lines that are only digits.
      - Lines starting with known phrases (e.g., "Under review").
      - Lines that consist solely of capital letters and spaces.
    """
    line_stripped = line.strip()
    if len(line_stripped) < 1:
        return True

    header_footer_patterns = [
        r"^\d+$",  # Only digits (e.g., page numbers like "000", "001", etc.)
        r"^Under review",  # Lines starting with "Under review"
    ]
    for pattern in header_footer_patterns:
        if re.match(pattern, line_stripped):
            return True
    return False


def clean_lines(content):
    """
    Given raw text content, split it into lines and remove lines that are
    likely headers/footers or otherwise not part of the main content.
    """
    lines = content.splitlines()
    # Keep only lines that are not detected as headers/footers.
    return [line for line in lines if not is_header_or_footer(line)]


def detect_references_position_clean(pdf_file):
    """
    Locate the first occurrence of the word "References" (or variations like
    "R EFERENCES") within the cleaned content extracted from the PDF.
    Uses pdftotext with layout preservation and cleans the extracted text.

    Returns a tuple (ref_page, ref_line) if found (with ref_line counting only
    the cleaned lines), otherwise None.
    """
    if not osp.exists(pdf_file):
        return None

    # Compile a regex pattern to match "REFERENCES" even if there are extra spaces
    # between letters (and do a case-insensitive match).
    pattern = re.compile(r"\bR\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S\b", re.IGNORECASE)

    # Loop through pages (limit to 50 pages by default)
    for page in range(1, 51):
        temp_dir = tempfile.mkdtemp()
        page_txt = osp.join(temp_dir, f"page_{page}.txt")
        try:
            subprocess.run(
                [
                    "pdftotext",
                    "-layout",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-q",
                    pdf_file,
                    page_txt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not osp.exists(page_txt):
                shutil.rmtree(temp_dir)
                break
            try:
                with open(page_txt, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
            except Exception as e:
                print(f"Error reading page {page}: {e}")
                print(traceback.format_exc())
                shutil.rmtree(temp_dir)
                continue
            finally:
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error running pdftotext for page {page}: {e}")
            print(traceback.format_exc())
            shutil.rmtree(temp_dir)
            continue

        # Clean the lines before searching for "References"
        cleaned = clean_lines(content)
        for idx, line in enumerate(cleaned):
            if pattern.search(line):
                # Found "References" on this page at cleaned line number idx+1
                return (page, idx + 1)
    return None


def extract_page_line_counts(pdf_file, first_page, last_page):
    """
    Extract the number of cleaned text lines for each page from first_page to last_page.
    This uses pdftotext with layout preservation and the clean_lines helper.
    Returns a dictionary {page_number: number_of_cleaned_lines}.
    Pages for which extraction fails are omitted.
    """
    page_lines = {}
    for page in range(first_page, last_page + 1):
        temp_dir = tempfile.mkdtemp()
        page_txt = osp.join(temp_dir, f"page_{page}.txt")
        try:
            subprocess.run(
                [
                    "pdftotext",
                    "-layout",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-q",
                    pdf_file,
                    page_txt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not osp.exists(page_txt):
                shutil.rmtree(temp_dir)
                break
            try:
                with open(page_txt, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
            except Exception as e:
                print(f"Error reading page {page}: {e}")
                print(traceback.format_exc())
                shutil.rmtree(temp_dir)
                continue
            finally:
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error running pdftotext for page {page}: {e}")
            print(traceback.format_exc())
            shutil.rmtree(temp_dir)
            continue
        # Clean the extracted text and count the number of remaining lines.
        cleaned = clean_lines(content)
        page_lines[page] = len(cleaned)
    return page_lines


def check_page_limit(pdf_file, page_limit=4, timeout=30):
    """
    Compile the LaTeX project in a temporary folder, then determine where the
    "References" section begins using cleaned text extraction. Next, count the
    number of cleaned text lines used before the word "References" and compare that
    to the total number of cleaned lines available in the allowed number of pages (page_limit).

    Returns a dictionary with:
      - 'ref_page': page number where "References" was found (or None)
      - 'ref_line': cleaned line number within that page (or None)
      - 'used_lines': number of cleaned lines used for main content (before "References")
      - 'allowed_lines': total number of cleaned text lines available in pages 1..page_limit
      - 'excess': if used_lines > allowed_lines (number of lines over the limit),
      - 'available': if used_lines < allowed_lines (number of lines still available)

    If compilation or extraction fails, returns None.
    """
    try:
        # Ensure the PDF was produced
        if not osp.exists(pdf_file):
            return None

        # Locate the first occurrence of "References" using the cleaned extraction
        ref_pos = detect_references_position_clean(pdf_file)
        if ref_pos is None:
            # If "References" isn't found, assume no reference section exists.
            return None
        ref_page, ref_line = ref_pos

        # Determine up to which page we need to extract cleaned line counts:
        max_page_to_extract = max(page_limit, ref_page)
        page_line_counts = extract_page_line_counts(pdf_file, 1, max_page_to_extract)
        if not page_line_counts:
            return None

        # Compute total cleaned lines available in the allowed pages (pages 1 to page_limit)
        allowed_lines = sum(
            page_line_counts.get(page, 0) for page in range(1, page_limit + 1)
        )

        # Compute cleaned lines used before "References":
        used_lines = 0
        # Sum full pages before the reference page
        for page in range(1, ref_page):
            used_lines += page_line_counts.get(page, 0)
        # Add lines from the reference page up to (but not including) the line where "References" appears
        used_lines += ref_line - 1

        result = {
            "ref_page": ref_page,
            "ref_line": ref_line,
            "used_lines": used_lines,
            "allowed_lines": allowed_lines,
        }
        if used_lines > allowed_lines:
            result["excess"] = used_lines - allowed_lines
        else:
            result["available"] = allowed_lines - used_lines
        return result

    except Exception as e:
        print(f"Error checking page limit: {e}")
        print(traceback.format_exc())
        return None


def get_reflection_page_info(reflection_pdf, page_limit):
    info = check_page_limit(reflection_pdf, page_limit)
    if info is not None:
        if "excess" in info:
            reflection_page_info = (
                f"\nCurrently, 'References' begins on page {info['ref_page']}, approximately on line {info['ref_line']}. "
                f"The main text (before the references) uses {info['used_lines']} lines, which exceeds the allowed {info['allowed_lines']} lines for a {page_limit}-page limit by {info['excess']} lines. "
                f"DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT. Please reduce the text or resize the plot to meet the page limit. "
                f"Consider grouping plots together to make the paper more concise. "
                f"Papers often look more professional if the main text is just under {page_limit} pages in length.\n"
            )
        elif "available" in info:
            reflection_page_info = (
                f"\nCurrently, 'References' begins on page {info['ref_page']}, approximately on line {info['ref_line']}. "
                f"The main text (before the references) uses {info['used_lines']} lines, leaving {info['available']} lines available out of the allowed {info['allowed_lines']} lines (which corresponds to {page_limit} pages). "
                f"DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT. You can add up to {info['available']} lines if needed, "
                f"but papers often look more professional if the main text is just under {page_limit} pages in length.\n"
            )
        else:
            # Fallback in case the info dictionary doesn't contain 'excess' or 'available'
            reflection_page_info = (
                f"\nCurrently, 'References' begins on page {info.get('ref_page', '?')}, approximately on line {info.get('ref_line', '?')}. "
                f"The page limit is {page_limit} pages for the main text before the references. "
                f"DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT. Adjust your content accordingly.\n"
            )
    else:
        reflection_page_info = (
            "\nCould not detect 'References' page (compilation or detection failed).\n"
        )

    return reflection_page_info


def get_citation_addition(
    client, model, context, current_round, total_rounds, idea_text
):
    report, citations = context
    msg_history = []
    citation_system_msg_template = """You are an ambitious AI researcher who is looking to publish a paper to a workshop at ICLR 2025 that explores real-world pitfalls, failures, and challenges in deep learning.
You have already completed the experiments and now you are looking to collect citations to related papers.
This phase focuses on collecting references and annotating them to be integrated later.
Collected citations will be added to a references.bib file.

Reasons to reference papers include:
1. Summarizing Research: Cite sources when summarizing the existing literature.
2. Using Specific Concepts: Provide citations when discussing specific theories or concepts.
3. Datasets, models, and optimizers: Cite the creators of datasets, models, and optimizers.
4. Comparing Findings: Cite relevant studies when comparing or contrasting different findings.
5. Highlighting Research Gaps: Cite previous research when pointing out gaps your study addresses.
6. Using Established Methods: Cite the creators of methodologies you employ.
7. Supporting Arguments: Cite sources that back up your conclusions and arguments.
8. Suggesting Future Research: Reference studies related to proposed future research directions.

Ensure sufficient cites will be collected for all of these categories, and no categories are missed.
You will be given access to the Semantic Scholar API; only add citations that you have found using the API.
Aim to discuss a broad range of relevant papers, not just the most popular ones.
Make sure not to copy verbatim from prior literature to avoid plagiarism.
You will have {total_rounds} rounds to add to the references but do not need to use them all.

DO NOT ADD A CITATION THAT ALREADY EXISTS!"""

    citation_first_prompt_template = """Round {current_round}/{total_rounds}:

You planned and executed the following idea:
```markdown
{Idea}
```

You produced the following report:
```markdown
{report}
```

Your current list of citations is:
```
{citations}
```

Identify the most important citation that you still need to add, and the query to find the paper.

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason and identify which citations are missing.
If no more citations are needed, add "No more citations needed" to your thoughts.
Do not add "No more citations needed" if you are adding citations this round.

In <JSON>, respond in JSON format with the following fields:
- "Description": The purpose of the desired citation and a brief description of what you are looking for.
- "Query": The search query to find the paper (e.g., attention is all you need).
This JSON will be automatically parsed, so ensure the format is precise."""

    citation_second_prompt_template = """Search has recovered the following articles:

{papers}

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, briefly reason over the search results and identify which citation(s) best fit your paper.
If none are appropriate or would contribute significantly to the write-up, add "Do not add any" to your thoughts.
Do not select papers that are already in the `references.bib` file, or if the same citation exists under a different name.

In <JSON>, respond in JSON format with the following fields:
- "Selected": A list of integer indices for the selected papers, for example [0, 1]. Do not use quotes for the indices, e.g. "['0', '1']" is invalid.
- "Description": Update the previous description of the citation(s) with the additional context. This should be a brief description of the work(s), their relevance, and where in a paper these should be cited.
This JSON will be automatically parsed, so ensure the format is precise."""

    try:
        text, msg_history = get_response_from_llm(
            prompt=citation_first_prompt_template.format(
                current_round=current_round + 1,
                total_rounds=total_rounds,
                Idea=idea_text,
                report=report,
                citations=citations,
            ),
            client=client,
            model=model,
            system_message=citation_system_msg_template.format(
                total_rounds=total_rounds
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "No more citations needed" in text:
            print("No more citations needed.")
            return None, True

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        query = json_output["Query"]
        papers = search_for_papers(query, result_limit=5)
    except Exception:
        print("EXCEPTION in get_citation_addition (initial search):")
        print(traceback.format_exc())
        return None, False

    if papers is None:
        print("No papers found.")
        return None, False

    paper_strings = []
    for i, paper in enumerate(papers):
        paper_strings.append(
            "{i}: {title}. {authors}. {venue}, {year}.\nAbstract: {abstract}".format(
                i=i,
                title=paper["title"],
                authors=paper["authors"],
                venue=paper["venue"],
                year=paper["year"],
                abstract=paper["abstract"],
            )
        )
    papers_str = "\n\n".join(paper_strings)

    try:
        text, msg_history = get_response_from_llm(
            prompt=citation_second_prompt_template.format(
                papers=papers_str,
                current_round=current_round + 1,
                total_rounds=total_rounds,
            ),
            client=client,
            model=model,
            system_message=citation_system_msg_template.format(
                total_rounds=total_rounds
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "Do not add any" in text:
            print("Do not add any.")
            return None, False

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        desc = json_output["Description"]
        selected_papers = str(json_output["Selected"])

        if selected_papers != "[]":
            selected_indices = []
            for x in selected_papers.strip("[]").split(","):
                x_str = x.strip().strip('"').strip("'")
                if x_str:
                    selected_indices.append(int(x_str))
            assert all([0 <= i < len(papers) for i in selected_indices]), (
                "Invalid paper index"
            )
            bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_indices]

            cleaned_bibtexs = []
            for bibtex in bibtexs:
                newline_index = bibtex.find("\n")
                cite_key_line = bibtex[:newline_index]
                cite_key_line = remove_accents_and_clean(cite_key_line)
                cleaned_bibtexs.append(cite_key_line + bibtex[newline_index:])
            bibtexs = cleaned_bibtexs

            bibtex_string = "\n".join(bibtexs)
        else:
            return None, False

    except Exception:
        print("EXCEPTION in get_citation_addition (selecting papers):")
        print(traceback.format_exc())
        return None, False

    references_format = """% {description}
{bibtex}"""

    references_prompt = references_format.format(bibtex=bibtex_string, description=desc)
    return references_prompt, False


writeup_system_message_template = """You are an ambitious AI researcher who is looking to publish a paper to the "I Can't Believe It's Not Better" (ICBINB) Workshop at ICLR 2025.
This workshop aims to highlight real-world pitfalls, challenges, and negative or inconclusive results in deep learning, encouraging open discussion.
You must accurately represent the results of the experiments.
The main paper is limited to {page_limit} pages in single-column format, not counting references. In general, try to use the available space and include all relevant information.
DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT.
MINIMIZE THE USAGE OF ITEMIZE OR ENUMERATE. ONLY USE THEM IF THEY ARE ABSOLUTELY NECESSARY AND CONTAIN SUBSTANTIAL INFORMATION.
Ensure that the tables and figures are correctly placed in a reasonable location and format.

- Do not change the overall style which is mandated by the conference. Keep to the current method of including the references.bib file.
- Do not remove the \\graphicspath directive or no figures will be found.
- Do not add `Acknowledgements` section to the paper.

Here are some tips for each section of the paper:

- **Title**:
  - Title should be catchy and informative. It should give a good idea of what the paper is about.
  - Try to keep it under 2 lines.

- **Abstract**:
  - Brief summary highlighting the nature of the challenge or pitfall explored.
  - Concise motivation of why this matters for real-world deployment.
  - This should be one continuous paragraph.

- **Introduction**:
  - Overview of the issue or challenge being explored.
  - Clearly state why this problem is important, especially for practical or real-world contexts.
  - Summarize your contributions or findings: they may include negative results, real-world pitfalls, unexpected behaviors, or partial improvements.

- **Related Work**:
  - Cite relevant papers or approaches that have tackled similar issues or have encountered similar pitfalls.
  - Compare and contrast with your own findings.

- **Background** (optional):
  - Provide necessary technical or domain-specific background if needed.

- **Method / Problem Discussion**:
  - Detail the problem context or the method if it is relevant to highlight the challenges faced.
  - If results are not strictly an improvement, discuss partial successes or lessons learned.

- **Experiments** (if applicable):
  - Present results truthfully according to the data you have. Negative, unexpected, or inconclusive findings are valid contributions for this workshop.
  - Include figures, tables, or real-world examples that illustrate the pitfalls.
  - Include up to 4 figures in the main text. All other figures should be in the appendix.

- **Conclusion**:
  - Summarize the main lessons learned or contributions.
  - Suggest next steps or future directions, highlighting how these insights can help the community avoid or overcome similar issues.

- **Appendix**:
  - Place for supplementary material that did not fit in the main paper.
  - Add more information and details (hyperparameters, algorithms, etc.) in the supplementary material.
  - Add more plots and tables in the supplementary material. Make sure that this information is not already covered in the main paper.
  - When checking for duplicate figures, be sure to also review their descriptions to catch cases where different figures convey the same information. For example, one figure might present aggregated training accuracy as a single line plot with a shaded standard deviation (e.g., aggregated_training_accuracy.png), while another (per_seed_training_accuracy.png) shows the same data as three separate line plots.

Ensure you are always writing good compilable LaTeX code. Common mistakes that should be fixed include:
- LaTeX syntax errors (unenclosed math, unmatched braces, etc.).
- Duplicate figure labels or references.
- Unescaped special characters: & % $ # _ {{ }} ~ ^ \\
- Proper table/figure closure.
- Do not hallucinate new citations or any results not in the logs.

Ensure proper citation usage:
- Always include references within \begin{{filecontents}}{{references.bib}} ... \end{{filecontents}}, even if they haven't changed from the previous round.
- Use citations from the provided references.bib content.
- Each section (especially Related Work) should have multiple citations.

When returning final code, place it in fenced triple backticks with 'latex' syntax highlighting.
"""

writeup_prompt = """Your goal is to write up the following idea:

```markdown
{idea_text}
```

We have the following experiment summaries (JSON):
```json
{summaries}
```

We also have a script used to produce the final plots (use this to see how the plots are generated and what names are used in the legend):
```python
{aggregator_code}
```
Please also consider which plots can naturally be grouped together as subfigures.

Available plots for the writeup (use these filenames):
```
{plot_list}
```

We also have VLM-based figure descriptions:
```
{plot_descriptions}
```

Your current progress on the LaTeX write-up is:
```latex
{latex_writeup}
```

Produce the final version of the LaTeX manuscript now, ensuring the paper is coherent, concise, and reports results accurately.
Return the entire file in full, with no unfilled placeholders!
This must be an acceptable complete LaTeX writeup, suitable for a 4-page single-column workshop paper.
Make sure to use the citations from the references.bib file.

Please provide the updated LaTeX code for 'template.tex', wrapped in triple backticks
with "latex" syntax highlighting, like so:

```latex
<UPDATED LATEX CODE>
```
"""


def load_idea_text(base_folder):
    """
    Load the idea text from the base folder.
    """
    idea_text = ""
    research_idea_path = osp.join(base_folder, "research_idea.md")
    if osp.exists(research_idea_path):
        with open(research_idea_path, "r") as f_idea:
            idea_text = f_idea.read()
    else:
        idea_md_path = osp.join(base_folder, "idea.md")
        if osp.exists(idea_md_path):
            with open(idea_md_path, "r") as f_idea:
                idea_text = f_idea.read()
    return idea_text


def load_exp_summaries(base_folder):
    """
    Load the experiment summaries from the base folder.
    """
    summary_files = [
        ("logs/0-run/baseline_summary.json", "BASELINE_SUMMARY"),
        ("logs/0-run/research_summary.json", "RESEARCH_SUMMARY"),
        ("logs/0-run/ablation_summary.json", "ABLATION_SUMMARY"),
    ]
    loaded_summaries = {}
    for fname, key in summary_files:
        path = osp.join(base_folder, fname)
        if osp.exists(path):
            try:
                with open(path, "r") as f:
                    loaded_summaries[key] = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: {fname} is not valid JSON. Using empty data for {key}."
                )
                loaded_summaries[key] = {}
        else:
            loaded_summaries[key] = {}
    return loaded_summaries


def filter_experiment_summaries(exp_summaries, step_name):
    if step_name == "citation_gathering":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "metric",
            "vlm_feedback_summary",
        }
    elif step_name == "writeup":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "metric",
            "code",
            "plot_analyses",
            "vlm_feedback_summary",
        }
    elif step_name == "plot_aggregation":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "plot_plan",
            "plot_code",
            "plot_analyses",
            "vlm_feedback_summary",
            "exp_results_npy_files",
        }
    else:
        raise ValueError(f"Invalid step name: {step_name}")

    filtered_summaries = {}
    for stage_name in exp_summaries.keys():
        if stage_name in {"BASELINE_SUMMARY", "RESEARCH_SUMMARY"}:
            filtered_summaries[stage_name] = {}
            for key in exp_summaries[stage_name].keys():
                if key in {"best node"}:
                    filtered_summaries[stage_name][key] = {}
                    for node_key in exp_summaries[stage_name][key].keys():
                        if node_key in node_keys_to_keep:
                            filtered_summaries[stage_name][key][node_key] = (
                                exp_summaries[stage_name][key][node_key]
                            )
        elif stage_name == "ABLATION_SUMMARY" and step_name == "plot_aggregation":
            filtered_summaries[stage_name] = {}
            for ablation_summary in exp_summaries[stage_name]:
                if ablation_summary is None:
                    continue
                filtered_summaries[stage_name][ablation_summary["ablation_name"]] = {}
                for node_key in ablation_summary.keys():
                    if node_key in node_keys_to_keep:
                        filtered_summaries[stage_name][
                            ablation_summary["ablation_name"]
                        ][node_key] = ablation_summary[node_key]
    return filtered_summaries


def gather_citations(base_folder, num_cite_rounds=20, small_model=os.getenv("LLM_SMALL_MODEL", "gpt-4o-2024-05-13")):
    """
    Gather citations for a paper, with ability to resume from previous progress.

    Args:
        base_folder: Path to project folder
        num_cite_rounds: Maximum number of citation gathering rounds
        small_model: Model to use for citation collection
        resume: Whether to try to resume from previous progress

    Returns:
        str: The gathered citations text, or None if failed
    """

    # Paths for storing progress
    citations_cache_path = osp.join(base_folder, "cached_citations.bib")
    progress_path = osp.join(base_folder, "citations_progress.json")

    # Initialize or load progress
    current_round = 0
    citations_text = ""

    if osp.exists(citations_cache_path) and osp.exists(progress_path):
        try:
            with open(citations_cache_path, "r") as f:
                citations_text = f.read()
            with open(progress_path, "r") as f:
                progress = json.load(f)
                current_round = progress.get("completed_rounds", 0)
            print(f"Resuming citation gathering from round {current_round}")
        except Exception as e:
            print(f"Error loading cached citations: {e}")
            print("Starting fresh")
            current_round = 0
            citations_text = ""

    try:
        # Load idea text and summaries
        idea_text = load_idea_text(base_folder)
        exp_summaries = load_exp_summaries(base_folder)
        filtered_summaries = filter_experiment_summaries(
            exp_summaries, step_name="citation_gathering"
        )
        filtered_summaries_str = json.dumps(filtered_summaries, indent=2)

        # Run small model for citation additions
        client, client_model = create_client(small_model)

        for round_idx in range(current_round, num_cite_rounds):
            try:
                context_for_citation = (filtered_summaries_str, citations_text)
                addition, done = get_citation_addition(
                    client,
                    client_model,
                    context_for_citation,
                    round_idx,
                    num_cite_rounds,
                    idea_text,
                )

                if done:
                    # Save final state before exiting
                    with open(citations_cache_path, "w") as f:
                        f.write(citations_text)
                    with open(progress_path, "w") as f:
                        json.dump(
                            {"completed_rounds": round_idx + 1, "status": "completed"},
                            f,
                        )
                    break

                if addition is not None:
                    # Simple check to avoid duplicating the same title
                    title_match = re.search(r" title = {(.*?)}", addition)
                    if title_match:
                        new_title = title_match.group(1).lower()
                        existing_titles = re.findall(
                            r" title = {(.*?)}", citations_text
                        )
                        existing_titles = [t.lower() for t in existing_titles]
                        if new_title not in existing_titles:
                            citations_text += "\n" + addition
                            # Save progress after each successful addition
                            with open(citations_cache_path, "w") as f:
                                f.write(citations_text)
                            with open(progress_path, "w") as f:
                                json.dump(
                                    {
                                        "completed_rounds": round_idx + 1,
                                        "status": "in_progress",
                                    },
                                    f,
                                )

            except Exception as e:
                print(f"Error in citation round {round_idx}: {e}")
                print(traceback.format_exc())
                # Save progress even if there's an error
                with open(citations_cache_path, "w") as f:
                    f.write(citations_text)
                with open(progress_path, "w") as f:
                    json.dump({"completed_rounds": round_idx, "status": "error"}, f)
                continue

        return citations_text if citations_text else None

    except Exception:
        print("EXCEPTION in gather_citations:")
        print(traceback.format_exc())
        return citations_text if citations_text else None


def perform_writeup(
    base_folder,
    citations_text=None,
    no_writing=False,
    num_cite_rounds=20,
    small_model=os.getenv("LLM_SMALL_MODEL", "gpt-4o-2024-05-13"),
    big_model=os.getenv("LLM_MODEL", "o1-2024-12-17"),
    n_writeup_reflections=3,
    page_limit=4,
):
    pdf_file = osp.join(base_folder, f"{osp.basename(base_folder)}.pdf")
    latex_folder = osp.join(base_folder, "latex")

    # Cleanup any previous latex folder and pdf
    if osp.exists(latex_folder):
        shutil.rmtree(latex_folder)
    if osp.exists(pdf_file):
        os.remove(pdf_file)

    # Remove any previous reflection PDFs
    for old_pdf in os.listdir(base_folder):
        if old_pdf.endswith(".pdf") and "reflection" in old_pdf:
            os.remove(osp.join(base_folder, old_pdf))

    try:
        idea_text = load_idea_text(base_folder)
        exp_summaries = load_exp_summaries(base_folder)
        filtered_summaries_for_writeup = filter_experiment_summaries(
            exp_summaries, step_name="writeup"
        )
        # Convert them to one big JSON string for context
        combined_summaries_str = json.dumps(filtered_summaries_for_writeup, indent=2)

        # Prepare a new fresh latex folder
        if not osp.exists(osp.join(latex_folder, "template.tex")):
            shutil.copytree(
                "ai_scientist/blank_icbinb_latex", latex_folder, dirs_exist_ok=True
            )

        writeup_file = osp.join(latex_folder, "template.tex")
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        # Gather plot filenames from figures/ folder
        figures_dir = osp.join(base_folder, "figures")
        plot_names = []
        if osp.exists(figures_dir):
            for fplot in os.listdir(figures_dir):
                if fplot.lower().endswith(".png"):
                    plot_names.append(fplot)

        # Load aggregator script to include in the prompt
        aggregator_path = osp.join(base_folder, "auto_plot_aggregator.py")
        aggregator_code = ""
        if osp.exists(aggregator_path):
            with open(aggregator_path, "r") as fa:
                aggregator_code = fa.read()
        else:
            aggregator_code = "No aggregator script found."

        if no_writing:
            compile_latex(latex_folder, pdf_file)
            return osp.exists(pdf_file)

        # If no citations provided, try to load from cache first
        if citations_text is None:
            citations_cache_path = osp.join(base_folder, "cached_citations.bib")
            if osp.exists(citations_cache_path):
                try:
                    with open(citations_cache_path, "r") as f:
                        citations_text = f.read()
                    print("Loaded citations from cache")
                except Exception as e:
                    print(f"Error loading cached citations: {e}")
                    citations_text = None

            # If still no citations, gather them
            if not citations_text:
                citations_text = gather_citations(
                    base_folder, num_cite_rounds, small_model
                )
                if citations_text is None:
                    print("Warning: Citation gathering failed")
                    citations_text = ""

        # Insert citations into template.tex
        if citations_text:
            with open(writeup_file, "r") as f:
                content = f.read()
            pattern_end = r"\end{filecontents}"
            content = content.replace(pattern_end, f"\n{citations_text}{pattern_end}")
            with open(writeup_file, "w") as f:
                f.write(content)

        # Generate VLM-based descriptions
        try:
            vlm_client, vlm_model = create_vlm_client(os.getenv("LLM_MODEL", "gpt-4o-2024-05-13"))
            desc_map = {}
            for pf in plot_names:
                ppath = osp.join(figures_dir, pf)
                if not osp.exists(ppath):
                    continue
                img_dict = {
                    "images": [ppath],
                    "caption": "No direct caption",
                }
                review_data = generate_vlm_img_review(img_dict, vlm_model, vlm_client)
                if review_data:
                    desc_map[pf] = review_data.get(
                        "Img_description", "No description found"
                    )
                else:
                    desc_map[pf] = "No description found"

            plot_descriptions_list = []
            for fname in plot_names:
                desc_text = desc_map.get(fname, "No description found")
                plot_descriptions_list.append(f"{fname}: {desc_text}")
            plot_descriptions_str = "\n".join(plot_descriptions_list)
        except Exception:
            print("EXCEPTION in VLM figure description generation:")
            print(traceback.format_exc())
            plot_descriptions_str = "No descriptions available."

        big_model_system_message = writeup_system_message_template.format(
            page_limit=page_limit
        )
        big_client, big_client_model = create_client(big_model)
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        combined_prompt = writeup_prompt.format(
            idea_text=idea_text,
            summaries=combined_summaries_str,
            aggregator_code=aggregator_code,
            plot_list=", ".join(plot_names),
            latex_writeup=writeup_text,
            plot_descriptions=plot_descriptions_str,
        )

        response, msg_history = get_response_from_llm(
            prompt=combined_prompt,
            client=big_client,
            model=big_client_model,
            system_message=big_model_system_message,
            print_debug=False,
        )

        latex_code_match = re.search(r"```latex(.*?)```", response, re.DOTALL)
        if not latex_code_match:
            return False
        updated_latex_code = latex_code_match.group(1).strip()
        with open(writeup_file, "w") as f:
            f.write(updated_latex_code)

        # Multiple reflection loops on the final LaTeX
        for i in range(n_writeup_reflections):
            with open(writeup_file, "r") as f:
                current_latex = f.read()

            # Check for unused or invalid figure references
            referenced_figs_temp = re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?{([^}]+)}", current_latex
            )
            used_figs = set(os.path.basename(fig) for fig in referenced_figs_temp)
            all_figs = set(plot_names)
            unused_figs = all_figs - used_figs
            invalid_figs = used_figs - all_figs

            # Save PDF with reflection trial number
            reflection_pdf = osp.join(
                base_folder, f"{osp.basename(base_folder)}_reflection{i + 1}.pdf"
            )
            # Compile current version before reflection
            print(f"[green]Compiling PDF for reflection {i + 1}...[/green]")
            compile_latex(latex_folder, reflection_pdf)

            review_img_cap_ref = perform_imgs_cap_ref_review(
                vlm_client, vlm_model, reflection_pdf
            )

            # Detect duplicate figures between main text and appendix
            analysis_duplicate_figs = detect_duplicate_figures(
                vlm_client, vlm_model, reflection_pdf
            )
            print(analysis_duplicate_figs)

            # Get reflection_page_info
            reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)

            check_output = os.popen(  # TODO: should prob use subprocess instead
                f"chktex {writeup_file} -q -n2 -n24 -n13 -n1"
            ).read()

            reflection_prompt = f"""
Now let's reflect and identify any issues (including but not limited to):
1) Are there any LaTeX syntax errors or style violations we can fix? Refer to the chktex output below.
2) Is the writing clear, and scientifically rigorous for a workshop focusing on real-world pitfalls?
3) Have we included all relevant details from the summaries without hallucinating?
4) Are there short sections (one or two sentences) that could be combined into a single paragraph?
5) Can we use more information and details (hyperparameters, unused figures, etc.) in the supplementary material? Only add information that is not already covered in the main paper.
6) The following figures are available in the folder but not used in the LaTeX: {sorted(unused_figs)}
7) The following figure references in the LaTeX do not match any actual file: {sorted(invalid_figs)}
{reflection_page_info}
chktex results:
```
{check_output}
```
8) Issues identified in the VLM reviews of the images, their captions, and related text discussions. Ensure each caption clearly matches its image content and that there is substantial discussion of each figure in the text.
VLM reviews:
```
{review_img_cap_ref}
```

9) Duplicate figures between main text and appendix. Make sure to remove the duplicate figures from the appendix.
```
{analysis_duplicate_figs}
```

Please provide a revised complete LaTeX in triple backticks, or repeat the same if no changes are needed.
Return the entire file in full, with no unfilled placeholders!
This must be an acceptable complete LaTeX writeup.
Do not hallucinate any details!
Ensure proper citation usage:
- Always include references within \begin{{filecontents}}{{references.bib}} ... \end{{filecontents}}, even if they haven't changed from the previous round.
- Use citations from the provided references.bib content.
"""

            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history[-1:],
                print_debug=False,
            )

            # 2nd run:
            reflection_code_match = re.search(
                r"```latex(.*?)```", reflection_response, re.DOTALL
            )
            if reflection_code_match:
                reflected_latex_code = reflection_code_match.group(1).strip()
                if reflected_latex_code != current_latex:
                    final_text = reflected_latex_code
                    cleanup_map = {
                        "</end": r"\\end",
                        "</begin": r"\\begin",
                        "’": "'",
                    }
                    for bad_str, repl_str in cleanup_map.items():
                        final_text = final_text.replace(bad_str, repl_str)
                    final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                    with open(writeup_file, "w") as fo:
                        fo.write(final_text)

                    compile_latex(latex_folder, reflection_pdf)
                else:
                    print(f"No changes in reflection step {i + 1}.")
                    break
            else:
                print(f"No valid LaTeX code block found in reflection step {i + 1}.")
                break
            # Get new reflection_page_info
            reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)
            review_img_selection = perform_imgs_cap_ref_review_selection(
                vlm_client, vlm_model, reflection_pdf, reflection_page_info
            )
            img_reflection_prompt = f"""Now let's reflect on
The following figures are currently used in the paper: {sorted(used_figs)}
The following figures are available in the folder but not used in the LaTeX: {sorted(unused_figs)}

{reflection_page_info}

The following is the VLM review on figures:

{review_img_selection}

Please review the figures and make the following changes:
1. For figures that do not add significant value to the paper, move them to the appendix
2. For figures that are not very informative or do not effectively communicate meaningful patterns, remove them entirely
3. For figures that does not contain subfigures and present sparse information, consider combining them with other related figures
4. Update all relevant text discussions to reflect any changes in figure placement or combinations
5. Enhance the scientific analysis of the remaining figures in the text - provide detailed, insightful discussions of their significance and findings

Please ensure all changes maintain scientific rigor and improve the paper's clarity and impact.
Be more aggressive with figure selection - move more figures to the appendix or group them together with other figures if the page limit is already exceeded.

If you believe you are done with reflection, simply say: "I am done"."""
            reflection_response, msg_history = get_response_from_llm(
                prompt=img_reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history[-1:],
                print_debug=False,
            )

            if "I am done" in reflection_response:
                print(
                    "LLM indicated it is done with reflections. Exiting reflection loop."
                )
                break

            reflection_code_match = re.search(
                r"```latex(.*?)```", reflection_response, re.DOTALL
            )
            if reflection_code_match:
                reflected_latex_code = reflection_code_match.group(1).strip()
                if reflected_latex_code != current_latex:
                    final_text = reflected_latex_code
                    cleanup_map = {
                        "</end": r"\\end",
                        "</begin": r"\\begin",
                        "’": "'",
                    }
                    for bad_str, repl_str in cleanup_map.items():
                        final_text = final_text.replace(bad_str, repl_str)
                    final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                    with open(writeup_file, "w") as fo:
                        fo.write(final_text)

                    compile_latex(latex_folder, reflection_pdf)
                else:
                    print(f"No changes in reflection step {i + 1}.")
                    break
            else:
                print(f"No valid LaTeX code block found in reflection step {i + 1}.")
                break

        # Final reflection on page limit
        # Save PDF with reflection

        # Get new reflection_page_info
        reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)

        final_reflection_prompt = """{reflection_page_info}
USE MINIMAL EDITS TO OPTIMIZE THE PAGE LIMIT USAGE."""
        reflection_response, msg_history = get_response_from_llm(
            prompt=final_reflection_prompt,
            client=big_client,
            model=big_client_model,
            system_message=big_model_system_message,
            msg_history=msg_history[-1:],
            print_debug=False,
        )

        reflection_pdf = osp.join(
            base_folder, f"{osp.basename(base_folder)}_reflection_final_page_limit.pdf"
        )
        # Compile current version before reflection
        print("[green]Compiling PDF for reflection final page limit...[/green]")

        print(f"reflection step {i + 1}")

        reflection_code_match = re.search(
            r"```latex(.*?)```", reflection_response, re.DOTALL
        )
        if reflection_code_match:
            reflected_latex_code = reflection_code_match.group(1).strip()
            if reflected_latex_code != current_latex:
                final_text = reflected_latex_code
                cleanup_map = {
                    "</end": r"\\end",
                    "</begin": r"\\begin",
                    "’": "'",
                }
                for bad_str, repl_str in cleanup_map.items():
                    final_text = final_text.replace(bad_str, repl_str)
                final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                with open(writeup_file, "w") as fo:
                    fo.write(final_text)

                compile_latex(latex_folder, reflection_pdf)
            else:
                print("No changes in reflection page step.")

        return osp.exists(reflection_pdf)

    except Exception:
        print("EXCEPTION in perform_writeup:")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform writeup for a project")
    parser.add_argument("--folder", type=str, help="Project folder", required=True)
    parser.add_argument("--no-writing", action="store_true", help="Only generate")
    parser.add_argument("--num-cite-rounds", type=int, default=20)
    parser.add_argument(
        "--model",
        type=str,
        default="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        choices=AVAILABLE_LLMS,
        help="Model to use for citation collection (small model).",
    )
    parser.add_argument(
        "--big-model",
        type=str,
        default="o1-2024-12-17",
        choices=AVAILABLE_LLMS,
        help="Model to use for final writeup (big model).",
    )
    parser.add_argument(
        "--writeup-reflections",
        type=int,
        default=3,
        help="Number of reflection steps for the final LaTeX writeup.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=4,
        help="Target page limit for the main paper (excluding references).",
    )
    args = parser.parse_args()

    try:
        success = perform_writeup(
            base_folder=args.folder,
            no_writing=args.no_writing,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model,
            big_model=args.big_model,
            n_writeup_reflections=args.writeup_reflections,
            page_limit=args.page_limit,
        )
        if not success:
            print("Writeup process did not complete successfully.")
    except Exception:
        print("EXCEPTION in main:")
        print(traceback.format_exc())
