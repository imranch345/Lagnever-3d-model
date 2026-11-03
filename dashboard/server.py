"""The project dashboard, with the 3D viewer inside it.

Research, Level 1 and 2a, evidence and documents in one page. One local server, standard
library only, bound to 127.0.0.1::

    python -m dashboard.server --scene datasets/processed/level1/real/s0029
        --model datasets/processed/level2a/models/joint.npz --research ~/code/lagnav-3d

Routes, on top of every viewer route (``viewer.server``):

``GET /``                    the dashboard;  ``GET /dashboard.js`` its script
``GET /viewer``              the 3D viewer page, shown in the dashboard's viewer tab
``GET /api/status``          everything the dashboard shows (``dashboard.status.collect``)
``GET /api/doc?id=...``      one whitelisted document, as Markdown
``GET /evidence/<name>``     one evidence file (images and Markdown, by bare name only)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

from dashboard.status import Roots, collect, doc_path, evidence_path
from language.layer import load_user_env
from viewer.server import STATIC as VIEWER_STATIC
from viewer.server import ViewerApp, _handler, build_app

__all__ = ["make_dashboard_server", "main"]

STATIC = Path(__file__).parent / "static"
PRODUCT = Path(__file__).resolve().parents[1]
CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".md": "text/markdown; charset=utf-8",
}


def make_dashboard_server(app: ViewerApp, roots: Roots, port: int = 8795) -> ThreadingHTTPServer:
    """A server for the dashboard and the viewer it embeds, on 127.0.0.1 only."""
    viewer_handler = _handler(app)

    class Handler(viewer_handler):  # type: ignore[valid-type,misc]
        server_version = "LagnavDashboard/0.1"

        def do_GET(self) -> None:  # noqa: N802 - http.server's naming
            """Dashboard routes first, then the viewer's."""
            path, _, query = self.path.partition("?")
            static = {
                "/": (STATIC / "index.html", "text/html; charset=utf-8"),
                "/dashboard.js": (STATIC / "dashboard.js", "text/javascript; charset=utf-8"),
                "/viewer": (VIEWER_STATIC / "index.html", "text/html; charset=utf-8"),
            }
            if path in static:
                file_path, content_type = static[path]
                self._send(HTTPStatus.OK, file_path.read_bytes(), content_type)
                return
            if path == "/api/status":
                self._json(collect(roots))
                return
            if path == "/api/doc":
                doc_id = parse_qs(query).get("id", [""])[0]
                found = doc_path(roots, doc_id)
                if found is None:
                    self._json({"error": "no such document"}, HTTPStatus.NOT_FOUND)
                    return
                self._send(HTTPStatus.OK, found.read_bytes(), CONTENT_TYPES[".md"])
                return
            if path.startswith("/evidence/"):
                found = evidence_path(roots.product, unquote(path.removeprefix("/evidence/")))
                if found is None:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send(HTTPStatus.OK, found.read_bytes(), CONTENT_TYPES[found.suffix.lower()])
                return
            super().do_GET()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the dashboard."""
    parser = argparse.ArgumentParser(description="Lagnav project dashboard.")
    parser.add_argument("--scene", type=Path, required=True, help="built scene for the viewer")
    parser.add_argument("--model", type=Path, default=None, help="joint shape model (optional)")
    parser.add_argument(
        "--research",
        type=Path,
        default=Path.home() / "code" / "lagnav-3d",
        help="checkout where Step 21 runs (defaults to ~/code/lagnav-3d)",
    )
    parser.add_argument("--port", type=int, default=8795)
    args = parser.parse_args(argv)
    loaded = load_user_env()
    if loaded:
        print(f"Loaded from ~/.lagnav/anthropic.env: {', '.join(sorted(loaded))}", flush=True)
    research = args.research if args.research.is_dir() else PRODUCT
    roots = Roots(product=PRODUCT, research=research)
    server = make_dashboard_server(build_app(args.scene, args.model), roots, args.port)
    print(f"Lagnav dashboard on http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
