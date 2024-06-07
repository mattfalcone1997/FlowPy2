import subprocess
import os

from shutil import (which,
                    copy)
from tempfile import (TemporaryDirectory,
                      NamedTemporaryFile)


class ProcessError(RuntimeError):
    def __init__(self, out: subprocess.CompletedProcess):
        self._out = out

    def __str__(self) -> str:
        message = (f"Errorcode {self._out.returncode}, process output:\n"
                   f"\n{self._out.stdout.decode('utf-8')}")
        return message


class latexError(ProcessError):
    pass


class dvipsError(ProcessError):
    pass


class eps2epsError(ProcessError):
    pass


class TikzBuilder:
    def __init__(self, input_fn,
                 preamble=None,
                 latex_cmd='pdflatex',
                 document_class=None,
                 doc_options=None,
                 dependent_files=None):

        if not os.path.isfile(input_fn):
            raise FileNotFoundError(f'{input_fn} not found')

        self._name = input_fn
        self._latex_cmd = latex_cmd

        if document_class is None:
            document_class = 'article'

        self._doc_fn = self._check_document_class(document_class)

        self._doc_options = ['preview',
                             f'class={document_class}']

        if doc_options is not None:
            self._doc_options.extend(doc_options)

        if preamble is not None:
            if os.path.isfile(preamble):
                with open(preamble, 'r') as f:
                    preamble = f.readlines()

            elif isinstance(preamble, str):
                preamble = preamble.splitlines()

            else:
                raise TypeError("preamble must be a file or a str")

        self._preamble = preamble + ['\n\\usepackage{tikz}\n']

        if dependent_files is None:
            dependent_files = []

        elif not isinstance(dependent_files, list):
            raise TypeError("Dependent files must be of type list")

        for file in dependent_files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"{file} does not exist")

        self._dependent_files = dependent_files

    def _check_document_class(self, doc_class):
        fn = doc_class + '.cls'
        s = subprocess.run(['kpsewhich', fn],
                           capture_output=True)

        if s.returncode != 0:
            raise FileNotFoundError("If document class must be installed "
                                    "or in the current directory")

        if os.path.isfile(fn):
            return fn
        else:
            return None

    def _create_temporary(self):
        cwd = os.getcwd()
        temp = TemporaryDirectory()

        copy(os.path.join(cwd, self._name),
             temp.name)

        if self._doc_fn is not None:
            copy(os.path.join(cwd, self._doc_fn),
                 temp.name)

        for fn in self._dependent_files:
            copy(os.path.join(cwd, fn),
                 temp.name)

        return cwd, temp

    def _build_document(self, cwd, temp):
        lines = []

        options = ",".join(self._doc_options)
        lines.append("\\documentclass[%s]{standalone}\n" % options)

        if self._preamble is not None:
            lines.extend(self._preamble)

        lines.append("\\begin{document}\n")
        lines.append("\t\\input{%s}\n" % os.path.basename(self._name))
        lines.append("\\end{document}\n")

        fn = NamedTemporaryFile(mode='w',
                                suffix='.tex',
                                dir=temp.name,
                                delete=False)

        fn.writelines(lines)

        fn.close()

        return fn.name

    def _run_latex(self, fn: str, dvi=False):

        dvi_option = '-output-format=dvi' if dvi else ""
        cmd = [self._latex_cmd,
               "-file-line-error",
               "-halt-on-error",
               "-interaction=nonstopmode",
               dvi_option,
               fn]

        out = subprocess.run(cmd,
                             stdin=subprocess.DEVNULL,
                             capture_output=True)

        t_prefix = os.path.splitext(fn)[0]

        if out.returncode != 0:
            raise latexError(out)

        if dvi:
            return t_prefix+'.dvi'
        else:
            return t_prefix+'.pdf'

    def _run_dvips(self, src, dst=None):

        if not which('dvips'):
            raise RuntimeError("dvips not found")

        if dst is None:
            dst = os.path.splitext(src)[0] + '.eps'
        cmds = ['dvips', '-E', src, '-o', dst]
        out = subprocess.run(cmds, capture_output=True)

        if out.returncode != 0:
            dvipsError(out)

        return dst

    def _run_eps2eps(self, src, dst=None):
        if not which('eps2eps'):
            raise RuntimeError("eps2eps not found")

        if dst is None:
            dst = NamedTemporaryFile(dir=os.path.split(src)[0],
                                     suffix='.eps',
                                     delete=False)
        cmds = ['eps2eps', src, dst.name]
        out = subprocess.run(cmds, capture_output=True)
        if out.returncode != 0:
            raise eps2epsError(out)

        return dst.name

    def to_eps(self, output_fn=None):

        if output_fn is None:
            output_fn = os.path.splitext(self._name)[0] + '.eps'
        output_fn = os.path.abspath(output_fn)

        cwd, temp = self._create_temporary()

        fn = self._build_document(cwd, temp)
        os.chdir(temp.name)

        dvi_fn = self._run_latex(fn, dvi=True)

        eps_fn = self._run_dvips(dvi_fn)

        eps_fn2 = self._run_eps2eps(eps_fn)

        copy(eps_fn2, output_fn)
        os.chdir(cwd)

        temp.cleanup()

    def to_pdf(self, output_fn=None):
        if output_fn is None:
            output_fn = os.path.splitext(self._name)[0] + '.eps'
        output_fn = os.path.abspath(output_fn)

        cwd, temp = self._create_temporary()

        fn = self._build_document(cwd, temp)
        os.chdir(temp.name)

        pdf_fn = self._run_latex(fn, dvi=False)

        copy(pdf_fn, output_fn)
        os.chdir(cwd)
        temp.cleanup()
