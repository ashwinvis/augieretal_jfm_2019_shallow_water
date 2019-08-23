#!/usr/bin/env python
import os
import re


def is_input(line):
    """
    Determines whether or not a read in line contains an uncommented out
    \input{} statement. Allows only spaces between start of line and
    '\input{}'.
    """
    # tex_input_re = r"""^\s*\\input{[^}]*}""" # input only
    tex_input_re = (
        r"""(^[^\%]*\\input{[^}]*})|(^[^\%]*\\include{[^}]*})"""
    )  # input or include
    return re.search(tex_input_re, line)


def get_input(line):
    """
    Gets the file name from a line containing an input statement.
    """
    tex_input_filename_re = r"""{[^}]*"""
    m = re.search(tex_input_filename_re, line)
    return m.group()[1:]


def combine_path(base_path, relative_ref):
    """
    Combines the base path of the tex document being worked on with the
    relate reference found in that document.
    """
    if base_path != "":
        os.chdir(base_path)
    # Handle if .tex is supplied directly with file name or not
    if relative_ref.endswith(".tex"):
        return os.path.abspath(os.path.join(base_path, relative_ref))
    else:
        return os.path.abspath(relative_ref) + ".tex"


def expand_file(base_file, current_path=None, include_bbl=False):
    """
    Recursively-defined function that takes as input a file and returns it
    with all the inputs replaced with the contents of the referenced file.
    """
    output_lines = []
    f = open(base_file, "r")
    if current_path is None:
        current_path = os.curdir

    for line in f:
        if is_input(line):
            new_base_file = combine_path(current_path, get_input(line))
            output_lines += expand_file(new_base_file, current_path, include_bbl)
            output_lines.append("\n")  # add a new line after each file input
        elif (
            include_bbl
            and line.startswith("\\bibliography")
            and (not line.startswith("\\bibliographystyle"))
        ):
            output_lines += bbl_file(base_file)
        else:
            output_lines.append(line)
    f.close()
    return output_lines


def bbl_file(base_file):
    """
    Return content of associated .bbl file
    """
    bbl_path = os.path.abspath(os.path.splitext(base_file)[0]) + ".bbl"
    return open(bbl_path).readlines()


def _main(base_file, output_file, include_bbl=False):
    """
    This "flattens" a LaTeX document by replacing all \input{X} lines w/ the
    text actually contained in X. See associated README.md for details.
    """
    current_path = os.path.split(base_file)[0]
    g = open(output_file, "w")
    g.write("".join(expand_file(base_file, current_path, include_bbl)))
    g.close()
    return None


try:
    import click

    @click.command()
    @click.argument("base_file", type=click.Path())
    @click.argument("output_file", type=click.Path())
    @click.option("--include_bbl/--no_bbl", default=False)
    def main(base_file, output_file, include_bbl=False):
        _main(base_file, output_file, include_bbl)


except ImportError:
    import argparse

    main = _main

    parser = argparse.ArgumentParser()
    parser.add_argument("base_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--include_bbl", action="store_true")
    args = parser.parse_args()

    if __name__ == "__main__":
        args.output_file = os.path.abspath(args.output_file)
        main(
            base_file=args.base_file,
            output_file=args.output_file,
            include_bbl=args.include_bbl,
        )
