#let std-bibliography = bibliography // Due to argument shadowing.

#let font-family = ("P052",)

#let font-family-mono = ("Iosevka",)

#let font-size = (
  tiny: 6pt,
  script: 8pt, // scriptsize
  footnote: 9pt, // footnotesize
  small: 10pt,
  normal: 11pt, // normalsize
  large: 12pt,
  Large: 14pt,
  LARGE: 17pt,
  huge: 20pt,
  Huge: 25pt,
)

#let h(body) = {
  set text(size: font-size.normal, weight: "regular")
  set block(above: 11.9pt, below: 11.7pt)
  body
}

#let h1(body) = {
  set text(size: font-size.large, weight: "bold")
  set block(above: 13pt, below: 13pt)
  body
}

#let h2(body) = {
  set text(size: font-size.normal, weight: "bold")
  set block(above: 11.9pt, below: 11.8pt)
  body
}

#let h3(body) = {
  set text(size: font-size.normal, weight: "regular")
  set block(above: 11.9pt, below: 11.7pt)
  body
}


#let join-authors(authors) = {
  return if authors.len() > 2 {
    authors.join(", ", last: ", and ")
  } else if authors.len() == 2 {
    authors.join(" and ")
  } else {
    authors.at(0)
  }
}

#let make-author(author, affls) = {
  let author-affls = if type(author.affl) == array {
    author.affl
  } else {
    (author.affl,)
  }

  let lines = author-affls
    .map(key => {
        let affl = affls.at(key)
        let affl-keys = ("department", "institution", "location")
        return affl-keys
          .map(key => {
              let value = affl.at(key, default: none)
              if key != "location" {
                return value
              }

              // Location and country on the same line.
              let country = affl.at("country", default: none)
              if country == none {
                return value
              } else if value == none {
                return country
              } else {
                return value + ", " + country
              }
            })
          .filter(it => it != none)
          .join("\n")
      })
    .map(it => emph(it))

  return block(
    spacing: 0em,
    {
      show par: set block(spacing: 5.5pt)
      text(size: font-size.normal)[*#author.name*]
      set par(justify: true, leading: 5pt, first-line-indent: 0pt)
      text(size: font-size.small)[#lines.join([\ ])]
    },
  )
}

#let make-email(author) = {
  let label = text(size: font-size.small, smallcaps(author.email))
  return block(
    spacing: 0em,
    {
      // Compensate difference between name and email font sizes (10pt vs 9pt).
      v(1pt)
      link("mailto:" + author.email, label)
    },
  )
}

#let make-authors(authors, affls) = {
  let cells = authors.map(it => (make-author(it, affls), make-email(it))).join()
  return grid(
    columns: (6fr, 4fr),
    align: (left + top, right + top),
    row-gutter: 12pt,  // Visually perfect.
    ..cells)
}

#let make-title(title, authors, affls, abstract, keywords) = {
  // 1. Title.
  v(31pt - (0.25in + 4.5pt))
  block(
    width: 100%,
    spacing: 0em,
    {
      set align(center)
      set block(spacing: 0em)
      text(size: 14pt, weight: "bold", title)
    },
  )

  // 2. Authors.
  v(23.6pt, weak: true)
  make-authors(authors, affls)
  // 3. Editors if exist.
  // Render abstract.
  v(28.8pt, weak: true)
  block(
    spacing: 0em,
    width: 100%,
    {
      set text(size: font-size.small)
      set par(leading: 0.51em) // Original 0.55em (or 0.45em?).
      align(
        center,
        text(size: font-size.large, weight: "bold", [*Abstract*]),
      )
      v(8.2pt, weak: true)
      pad(left: 20pt, right: 20pt, abstract)
    },
  )

  // Render keywords if exist.
  if keywords != none {
    keywords = keywords.join([, ])
    v(6.5pt, weak: true) // ~1ex
    block(
      spacing: 0em,
      width: 100%,
      {
        set text(size: 10pt)
        set par(leading: 0.51em) // Original 0.55em (or 0.45em?).
        pad(left: 20pt, right: 20pt)[*Keywords:* #keywords]
      },
    )
  }

  // Space before paper content.
  v(23pt, weak: true)
}

#let jmlr(
  title: [],
  short-title: none,
  authors: (),
  last-names: (),
  date: auto,
  abstract: [],
  keywords: (),
  bibliography: none,
  appendix: none,
  body,
) = {
  // If there is no short title then use title as a short title.
  if short-title == none {
    short-title = title
  }

  // Authors are actually a tuple of authors and affilations.
  let affls = ()
  if authors.len() == 2 {
    (authors, affls) = authors
  }

  // If last names are not specified then try to guess last names from author
  // names.
  if last-names.len() == 0 and authors.len() > 0 {
    last-names = authors.map(it => it.name.trim("\s").split(" ").at(-1))
  }

  // Set document metadata.
  let meta-authors = join-authors(authors.map(it => it.name))
  set document(title: title, author: meta-authors, keywords: keywords, date: date)

  set page(
    paper: "us-letter",
    margin: (left: 1.0in, right: 1.0in, top: 1.0in + 4.0pt, bottom: 1in),
    header-ascent: 24pt + 0.25in + 4.5pt,
    footer-descent: 10%,
    footer: locate(loc => {
      let pageno = counter(page).at(loc).first()
      set text(size: font-size.script)
      grid(
        columns: (1fr, 1fr),
        align: (left, right),
        [], [#pageno],
      )
    }),
  )

  // Basic paragraph and text settings.
  set text(font: font-family, size: font-size.normal)
  set par(leading: 0.55em, first-line-indent: 17pt, justify: true)
  show par: set block(spacing: 0.55em)

  // Configure heading appearence and numbering.
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    show: h1
    // Render section with such names without numbering as level 3 heading.
    let unnumbered = (
      [Acknowledgments],
      [Acknowledgments and Disclosure of Funding],
    )
    if unnumbered.any(name => name == it.body) {
      set align(left)
      set text(size: font-size.large, weight: "bold")
      set par(first-line-indent: 0pt)
      v(0.3in, weak: true)
      block(spacing: 0pt, it.body)
      v(0.2in, weak: true)
    } else {
      it
    }
  }
  show heading.where(level: 2): h2
  show heading.where(level: 3): h3

  set enum(indent: 14pt, spacing: 15pt)
  show enum: set block(spacing: 18pt)

  set list(indent: 14pt, spacing: 15pt)
  show list: set block(spacing: 18pt)

  // Configure equations.
  // set math.equation(numbering: "(1)")
  show math.equation: set block(below: 8pt, above: 9pt)
  show math.equation: set text(weight: 400)


  show ref: it => {
    let eq = math.equation
    let el = it.element
    if el != none and el.func() == eq {
      numbering(el.numbering, ..counter(eq).at(el.location()))
    } else {
      it
    }
  }
  set cite(form: "prose")

  set figure(gap: 14pt)
  show figure.caption: it => {
    set text(size: font-size.small)
    set par(leading: 6.67pt, first-line-indent: 0pt)
    let numb = locate(loc => numbering(it.numbering, ..it.counter.at(loc)))
    let index = it.supplement + [~] + numb + it.separator
    grid(
      columns: 2,
      column-gutter: 5pt,
      align: left,
      index, it.body,
    )
  }

  make-title(title, authors, affls, abstract, keywords)
  parbreak()
  body

  if appendix != none {
    set heading(numbering: "A.1.", supplement: [Appendix])
    show heading: it => {
      let rules = (h1, h2, h3)
      let rule = rules.at(it.level - 1, default: h)
      show: rule
      let numb = locate(loc => {
        let counter = counter(heading)
        return numbering(it.numbering, ..counter.at(loc))
      })
      block([Appendix~#numb~#it.body])
    }

    counter(heading).update(0)
    pagebreak()
    appendix
  }

  if bibliography != none {
    show heading: it => {
      show: h1
      block(above: 0.32in, it.body)
    }
    // TODO(@daskol): Closest bibliography style is "bristol-university-press".
    set std-bibliography(
      title: [References],
      style: "bristol-university-press",
    )
    bibliography
  }
}

// The ASM template also provides a theorem function.
#let theorem-counter = counter("theorem")
#let theorem(body, numbered: true) = locate(location => {
  let lvl = counter(heading).at(location)
  let i = theorem-counter.at(location).first()
  let top = if lvl.len() > 0 { lvl.first() } else { 0 }
  show: block.with(spacing: 11.5pt)
  strong({
    [Theorem]
    if numbered [ #top.#i] + [.]
  })
  [ ]
  emph(body)
})

// And a function for a proof.
#let proof(body) = block(
  spacing: 11.5pt,
  {
    emph[Proof.]
    [ ] + body
    h(1fr)
    box(scale(160%, origin: bottom + right, sym.square.stroked))
  },
)
