# spec_xml_adapter.py
from typing import Dict, Any, List, Tuple
from lxml import etree

class SpecXMLAdapter:
#YAML mapping

    def __init__(self, mapping: Dict[str, Any], nsmap: Dict[str, str] | None = None):
        self.m = mapping
        self.ns = nsmap or {}

    def _str_first(self, el, xp: str) -> str:
        vals = el.xpath(xp, namespaces=self.ns)
        if not vals:
            return ""
        v = vals[0]
        return (v if isinstance(v, str) else getattr(v, "text", "") or "").strip()

    def _str_join(self, el, xp: str) -> str:
        vals = el.xpath(xp, namespaces=self.ns)
        out: List[str] = []
        for v in vals:
            if isinstance(v, str):
                out.append(v.strip())
            else:
                out.append((getattr(v, "text", "") or "").strip())
        return " ".join([s for s in out if s])

    def parse(self, xml_path: str) -> Tuple[List[dict], List[Tuple[str, str, dict]]]:
        tree = etree.parse(xml_path)
        root = tree.getroot()

        # Nodes
        ncfg = self.m["nodes"]
        node_elems = root.xpath(ncfg["select"], namespaces=self.ns)
        nodes = []
        for el in node_elems:
            nid   = self._str_first(el, ncfg["id"])
            ntype = self._str_first(el, ncfg.get("type", "string('Unknown')"))
            attrs = {}
            for k, xp in (ncfg.get("attrs") or {}).items():
                attrs[k] = self._str_join(el, xp)
            attrs.setdefault("name", attrs.get("name", nid))
            attrs.setdefault("documentation", attrs.get("documentation", ""))
            nodes.append({"id": nid, "type": ntype, **attrs})

        # Edges
        ecfg = self.m["edges"]
        edge_elems = root.xpath(ecfg["select"], namespaces=self.ns)
        edges = []
        for el in edge_elems:
            src  = self._str_first(el, ecfg["src"])
            dst  = self._str_first(el, ecfg["dst"])
            etyp = self._str_first(el, ecfg.get("type", "string('related_to')")) or "related_to"
            attrs = {}
            for k, xp in (ecfg.get("attrs") or {}).items():
                attrs[k] = self._str_join(el, xp)
            attrs["type"] = etyp
            edges.append((src, dst, attrs))

        # drop edges to missing endpoints
        ids = {n["id"] for n in nodes}
        edges = [(s, d, a) for (s, d, a) in edges if s in ids and d in ids]
        return nodes, edges
