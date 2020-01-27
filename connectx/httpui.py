import http.server
import threading
import cgi
import urllib


def format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class HTTPHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, config_manager, request, client_address, server):
        self.config_manager = config_manager
        super().__init__(request, client_address, server)

    def redirect(self, location):
        self.send_response(301)
        self.send_header("Location", location)
        self.end_headers()
        self.wfile.write(b"Moved")

    def handle_get(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=UTF-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        config = self.config_manager.current_config()

        rows = []
        for k, v in config.items():
            v = format_value(v)
            rows.append(f"<div>{k} <input type='text' name='{k}' value='{v}'></div>")

        rows.append("<input type='submit' value='Push'>")
        content = "".join(rows)

        form = f"<form action='/update' method='post'>{content}</form>"

        page = f"<html><head><title>RL Lab</title></head><body>{form}</body></html>"

        self.wfile.write(page.encode("utf8"))

    def handle_post(self):
        ctype, pdict = cgi.parse_header(self.headers["content-type"])
        if ctype == "multipart/form-data":
            postvars = cgi.parse_multipart(self.rfile, pdict)
        elif ctype == "application/x-www-form-urlencoded":
            length = int(self.headers["content-length"])
            postvars = urllib.parse.parse_qs(
                self.rfile.read(length), keep_blank_values=1
            )
        else:
            postvars = {}

        config = self.config_manager.current_config()

        for k, v in postvars.items():
            config[k.decode("utf8")] = v[0].decode("utf8")

        print(config)

        updated, message = self.config_manager.push_update(config)

        self.redirect(f"/?message={message}")

    def do_GET(self):
        self.handle_get()

    def do_POST(self):
        self.handle_post()


def run_http_server(config_manager):
    def create_handler(request, client_address, server):
        return HTTPHandler(config_manager, request, client_address, server)

    def start():
        server_address = ("", 8080)
        httpd = http.server.ThreadingHTTPServer(server_address, create_handler)
        httpd.serve_forever()

    t = threading.Thread(target=start, daemon=True)
    t.start()
