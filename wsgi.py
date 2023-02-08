"""
WSGI start file für die Anwendung.

Dieses Modul wird z.B. von gunicorn aufgerufen, um die Anwendung zu starten.
"""

from app import server as application


if __name__ == '__main__':
    application.run()
