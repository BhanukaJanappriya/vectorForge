"""SVG assembly: serialize a VectorDocument into a clean, optimized SVG 1.1 string.

Stage entrypoint: :func:`assemble_svg` (see ``contracts/stages.py``).

Output structure
----------------
* Root ``<svg version="1.1">`` in the SVG namespace with ``width``/``height`` (source px) and
  ``viewBox="0 0 W H"``.
* A ``<metadata>`` block holding a ``vf:document`` element (generator, schema version, source
  file, image class, palette) and the conversion settings as JSON.
* One ``<g>`` per VectorLayer, bottom to top in z-order, with ``id=layer.id``,
  ``inkscape:groupmode="layer"`` and ``inkscape:label=layer.name`` so both Inkscape and
  Illustrator show named layers. Paint lives on the group: ``fill`` + ``fill-rule`` for fills,
  or ``fill="none"`` + ``stroke``/``stroke-width`` with round joins and caps for strokes.
* No scripts, ``foreignObject``, external references, or embedded rasters are ever produced.

Optimization
------------
Every coordinate is first rounded to :data:`COORD_DECIMALS` decimals in absolute space (so the
error is bounded by 0.005 px and does not accumulate). The document is then run through scour
with a significant-digit precision large enough to represent every rounded absolute coordinate
exactly, so scour's relative/shorthand path rewriting is lossless with respect to the rounded
geometry. Layer ids, Inkscape layer attributes and the metadata block are protected.
"""

from __future__ import annotations

import json
import re

from lxml import etree
from scour import scour

from contracts.schemas import SCHEMA_VERSION, Settings, StageError, VectorDocument, VectorLayer

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
VF_NS = "https://vectorforge.dev/ns/svg/1"
"""Namespace of the VectorForge metadata element inside ``<metadata>``."""

COORD_DECIMALS = 2
"""Decimal places kept for path coordinates in output.svg."""

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_STAGE = "assemble"


def format_number(value: float, decimals: int = COORD_DECIMALS) -> str:
    """Format ``value`` rounded to ``decimals`` places without trailing zeros (``-0`` -> ``0``)."""
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def round_path(d: str, decimals: int = COORD_DECIMALS) -> tuple[str, float]:
    """Round every number in path data ``d`` to ``decimals`` places.

    Commands and their order are preserved, so the result is still an absolute M/L/C/Z path.

    Returns:
        The rounded path string and the largest absolute coordinate value it contains.
    """
    largest = 0.0

    def repl(match: re.Match[str]) -> str:
        nonlocal largest
        value = float(match.group())
        largest = max(largest, abs(value))
        return format_number(value, decimals)

    # Only number tokens are rewritten, so separators and commands are kept as they were.
    return _NUM_RE.sub(repl, d), largest


def _set_paint(group: etree._Element, layer: VectorLayer) -> None:
    """Put the layer's paint attributes on its ``<g>``."""
    if layer.is_stroke:
        width = max(float(layer.stroke_width or 0.0), 10.0**-COORD_DECIMALS)
        group.set("fill", "none")
        group.set("stroke", layer.color_hex)
        group.set("stroke-width", format_number(width))
        group.set("stroke-linejoin", "round")
        group.set("stroke-linecap", "round")
    else:
        group.set("fill", layer.color_hex)
        group.set("fill-rule", layer.fill_rule)
    if layer.opacity < 1.0:
        group.set("opacity", format_number(layer.opacity, 3))


def _metadata(root: etree._Element, doc: VectorDocument, settings: Settings) -> None:
    """Append the ``<metadata>`` block (generator, schema version, settings JSON)."""
    meta = etree.SubElement(root, f"{{{SVG_NS}}}metadata", id="vectorforge-metadata")
    info = etree.SubElement(meta, f"{{{VF_NS}}}document", nsmap={"vf": VF_NS})
    info.set("generator", doc.metadata.generator)
    info.set("schema-version", SCHEMA_VERSION)
    info.set("source", doc.metadata.source_filename)
    info.set("image-class", str(doc.metadata.image_class))
    info.set("palette", " ".join(doc.metadata.palette_hex))
    settings_el = etree.SubElement(info, f"{{{VF_NS}}}settings")
    settings_el.set("format", "application/json")
    settings_el.text = json.dumps(settings.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def build_svg(doc: VectorDocument, settings: Settings, *, optimize: bool = True) -> str:
    """Serialize ``doc`` to SVG 1.1.

    Args:
        doc: Document to serialize (paths in source space).
        settings: Settings recorded in the metadata block.
        optimize: When False, paths are written verbatim and scour is skipped (reference output
            for measuring what optimization changes).

    Returns:
        The SVG document as a string with an XML declaration.
    """
    root = etree.Element(f"{{{SVG_NS}}}svg", nsmap={None: SVG_NS, "inkscape": INKSCAPE_NS})
    root.set("version", "1.1")
    root.set("width", str(doc.width))
    root.set("height", str(doc.height))
    root.set("viewBox", f"0 0 {doc.width} {doc.height}")
    _metadata(root, doc, settings)

    largest = float(max(doc.width, doc.height))
    for layer in doc.layers:
        group = etree.SubElement(root, f"{{{SVG_NS}}}g", id=layer.id)
        group.set(f"{{{INKSCAPE_NS}}}groupmode", "layer")
        group.set(f"{{{INKSCAPE_NS}}}label", layer.name)
        _set_paint(group, layer)
        for d in layer.paths:
            if optimize:
                d, path_max = round_path(d)
                largest = max(largest, path_max)
            etree.SubElement(group, f"{{{SVG_NS}}}path", d=d)

    raw = etree.tostring(root, xml_declaration=True, encoding="UTF-8").decode("utf-8")
    if not optimize:
        return raw
    optimized = optimize_svg(raw, significant_digits=len(str(int(largest))) + COORD_DECIMALS)
    return _restore_root_size(optimized, doc)


_ROOT_TAG_RE = re.compile(r"<svg\s[^>]*>")
_SIZE_ATTR_RE = re.compile(r'\s(width|height|viewBox)="[^"]*"')


def _restore_root_size(svg: str, doc: VectorDocument) -> str:
    """Rewrite root ``width``/``height``/``viewBox`` as plain integers.

    scour shortens numbers to scientific notation (``2000`` -> ``2e3``). That is valid SVG,
    but some importers (notably older Illustrator versions) mis-read it on the root element.
    """
    match = _ROOT_TAG_RE.search(svg)
    if match is None:
        raise StageError(_STAGE, "optimized SVG has no <svg> root tag")
    values = {"width": str(doc.width), "height": str(doc.height), "viewBox": f"0 0 {doc.width} {doc.height}"}
    tag = _SIZE_ATTR_RE.sub(lambda m: f' {m.group(1)}="{values[m.group(1)]}"', match.group())
    return svg[: match.start()] + tag + svg[match.end() :]


def _scour_options(significant_digits: int) -> object:
    """Conservative scour options: lossless paths, keep ids, layers and metadata."""
    options = scour.sanitizeOptions()
    options.digits = significant_digits
    options.cdigits = significant_digits
    options.keep_editor_data = True  # inkscape:groupmode / inkscape:label
    options.strip_ids = False
    options.shorten_ids = False
    options.group_collapse = False
    options.group_create = False
    options.remove_metadata = False
    options.remove_descriptive_elements = False
    options.enable_viewboxing = False  # keep explicit width/height
    options.strip_comments = True
    options.strip_xml_prolog = False
    options.indent_type = "none"
    options.newlines = True
    options.quiet = True
    return options


def optimize_svg(svg: str, significant_digits: int) -> str:
    """Run scour over ``svg`` with enough precision to keep every coordinate exact.

    Args:
        svg: SVG whose coordinates are already rounded to :data:`COORD_DECIMALS` decimals.
        significant_digits: Digits needed for the largest absolute coordinate (integer digits
            plus :data:`COORD_DECIMALS`).
    """
    return str(scour.scourString(svg, _scour_options(significant_digits)))


def svg_problems(svg: str, doc: VectorDocument) -> list[str]:
    """Structural problems of an assembled SVG string (empty list = valid).

    Checks well-formedness, SVG namespace root, ``viewBox``/``width``/``height``, the layer
    groups (ids, Inkscape layer attributes, z-order) and forbidden content (scripts,
    ``foreignObject``, external references).
    """
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)
        root = etree.fromstring(svg.encode("utf-8"), parser)
    except etree.XMLSyntaxError as exc:
        return [f"not well-formed XML: {exc}"]
    problems: list[str] = []
    if root.tag != f"{{{SVG_NS}}}svg":
        problems.append(f"root is {root.tag!r}, expected svg in the SVG namespace")
    if root.get("viewBox") != f"0 0 {doc.width} {doc.height}":
        problems.append(f"viewBox {root.get('viewBox')!r} != '0 0 {doc.width} {doc.height}'")
    if (root.get("width"), root.get("height")) != (str(doc.width), str(doc.height)):
        problems.append(f"width/height {root.get('width')}x{root.get('height')} != {doc.width}x{doc.height}")
    groups = [el for el in root.iter(f"{{{SVG_NS}}}g") if el.get(f"{{{INKSCAPE_NS}}}groupmode") == "layer"]
    got = [(g.get("id"), g.get(f"{{{INKSCAPE_NS}}}label")) for g in groups]
    want = [(layer.id, layer.name) for layer in doc.layers]
    if got != want:
        problems.append(f"layer groups {got} != expected {want}")
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        local = etree.QName(el).localname
        if local in ("script", "foreignObject", "image", "use"):
            problems.append(f"forbidden element <{local}>")
        for name, value in el.attrib.items():
            attr = etree.QName(name).localname
            if attr == "href" or attr.startswith("on") or ("url(" in value and not value.startswith("url(#")):
                problems.append(f"forbidden attribute {attr}={value[:40]!r} on <{local}>")
    return problems


def assemble_svg(doc: VectorDocument, settings: Settings) -> str:
    """Serialize ``doc`` to an optimized, valid SVG 1.1 string with one layer ``<g>`` per VectorLayer.

    Raises:
        StageError: if serialization or optimization produced an invalid document.
    """
    try:
        svg = build_svg(doc, settings, optimize=True)
    except (ValueError, etree.LxmlError) as exc:  # pragma: no cover - defensive, lxml/scour internals
        raise StageError(_STAGE, f"SVG serialization failed: {exc}") from exc
    problems = svg_problems(svg, doc)
    if problems:
        raise StageError(_STAGE, "optimized SVG is invalid: " + "; ".join(problems))
    return svg


__all__: list[str] = [
    "COORD_DECIMALS",
    "assemble_svg",
    "build_svg",
    "format_number",
    "optimize_svg",
    "round_path",
    "svg_problems",
]
