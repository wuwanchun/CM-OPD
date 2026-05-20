# EMNLP LaTeX Draft Notes

This directory contains an initial EMNLP/ACL-style LaTeX draft for:

```text
Learning When to Expand: Progressive Evidence Disclosure for Long-Context Reasoning
```

Main files:

```text
progressive-evidence-disclosure-emnlp.tex
progressive-evidence-disclosure.bib
```

## Official ACL Style Files

EMNLP uses the ACL style files. The official open-source repository is:

```text
https://github.com/acl-org/acl-style-files
```

Before compiling, place the following official files in this same directory:

```text
acl.sty
acl_natbib.bst
```

If network access is available:

```bash
curl -L -O https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty
curl -L -O https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl_natbib.bst
```

Then compile with:

```bash
latexmk -pdf progressive-evidence-disclosure-emnlp.tex
```

All empirical results in the current draft are placeholders marked as `TBD`.

