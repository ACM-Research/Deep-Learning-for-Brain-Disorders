(TeX-add-style-hook
 "MODELS"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem") ("biblatex" "backend=biber" "style=ieee")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "inputenc"
    "fontenc"
    "graphicx"
    "grffile"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "amssymb"
    "capt-of"
    "hyperref"
    "biblatex")
   (LaTeX-add-labels
    "sec:orgfb067f8"
    "sec:orgf3c3aa0"
    "sec:org24d4053"
    "sec:org83a7fb6"
    "sec:org7bd9cd8"
    "sec:orga6a0d57"
    "sec:orgeb0ad26"
    "sec:org257aa70"
    "sec:org5348284"
    "sec:orge09e07d"
    "sec:orgce44384"
    "sec:orgb485094"
    "sec:org8dabafa"
    "sec:orgcad6efa")
   (LaTeX-add-bibliographies))
 :latex)

