import http.server, socketserver
class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json'}
socketserver.TCPServer.allow_reuse_address = True
socketserver.TCPServer(("127.0.0.1", 5322), H).serve_forever()
