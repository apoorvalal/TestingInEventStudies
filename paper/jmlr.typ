#let jmlr(
  title: [],
  authors: (),
  abstract: [],
  keywords: (),
  bibliography: none,
  appendix: none,
  date: none,
  body,
) = {
  // Extract affls if provided in the specific format
  let affls = ()
  if authors.len() == 2 and type(authors) == array {
    (authors, affls) = authors
  }

  // Basic document setup
  set document(title: title)
  set page(
    paper: "us-letter",
    margin: (left: 1.0in, right: 1.0in, top: 1.0in, bottom: 1.0in),
    numbering: "1",
  )

  // Basic text settings
  set text(font: ("P052",), size: 11pt)
  set par(leading: 0.55em, first-line-indent: 17pt, justify: true)
  set heading(numbering: "1.1  ")

  // Set citation style to dark red
  // set cite(style: "chicago-author-date")
  show cite: set text(fill: rgb(139, 0, 0))  // Dark red color for citations

  // Make all links dark red with no underline
  show link: set text(fill: rgb(139, 0, 0))

  // Title
  align(center)[
    #block(text(size: 14pt, weight: "bold", title))
    #v(1em)
  ]

  // Authors
  for author in authors {
    align(center)[
      #text(weight: "bold", author.name)
      #if "affl" in author and author.affl in affls {
        let affl = affls.at(author.affl)
        if "department" in affl {
          linebreak()
          emph(affl.department)
        }
      }
      #if "email" in author and author.email != "" {
        linebreak()
        link("mailto:" + author.email, author.email)
      }
    ]
    v(0.5em)
  }

  // Abstract
  if abstract != [] {
    v(1em)
    align(center)[*Abstract*]
    block(
      width: 100%,
      inset: (x: 2em),
      abstract
    )
  }

  // Keywords
  if keywords != () {
    v(0.5em)
    block(
      width: 100%,
      inset: (x: 2em),
      [*Keywords:* #keywords.join(", ")]
    )
  }

  v(2em)

  // Main body
  body

  // Appendix
  if appendix != none {
    pagebreak()
    heading(numbering: "A.1", [Appendix])
    counter(heading).update(0)
    appendix
  }

  // Bibliography
  if bibliography != none {
    pagebreak()
    heading([References])
    bibliography
  }
}

// Simplest possible theorem function
#let theorem(body) = {
  block(
    fill: rgb(240, 240, 240),
    inset: 1em,
    radius: 4pt,
    [*Theorem.* #body]
  )
}

// Simplest possible proof function
#let proof(body) = {
  block(
    inset: 1em,
    [*Proof.* #body #h(1fr) #sym.square.stroked]
  )
}
