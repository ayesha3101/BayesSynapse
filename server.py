"""
Run BayeSynapse on your own computer (no Vercel needed):

    python server.py

then open http://localhost:8000 in your browser.
It serves index.html and answers /api/solve with the same Python code Vercel uses.
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "api"))
from solve import handler as Api  # the same handler Vercel runs

PORT = 8000


class Local(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)  # serve files from this folder

    def do_POST(self):
        if self.path == "/api/solve":
            Api.do_POST(self)  # reuse the solver's request handling
        else:
            self.send_error(404)

    def log_message(self, *args):  # keep the terminal quiet
        pass


if __name__ == "__main__":
    print(f"BayeSynapse is running at http://localhost:{PORT}  (press Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Local).serve_forever()
