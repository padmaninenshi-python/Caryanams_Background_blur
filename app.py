"""
Standalone Remove Background + Number Plate Tool
Local run : python app.py
Render run: gunicorn app:app   (see Procfile)
"""

import os
from flask import Flask
from extensions import db


def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', 'removebg-standalone-secret')

    # SQLite DB
    db_path = os.path.join(app.root_path, 'instance', 'removebg.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    # Allow canvas crossOrigin reads for mirror reflection (same-origin images)
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        return response

    from routes import bp
    app.register_blueprint(bp)

    with app.app_context():
        db.create_all()

    return app


# ── Module-level app object ─────────────────────────────────────────────────
# Required so gunicorn can be started with "gunicorn app:app" (see Procfile).
# This also runs once when the worker process boots — same on Render & locally.
app = create_app()


if __name__ == '__main__':
    # Local dev only. On Render, gunicorn (see Procfile) starts the app instead
    # and this block is never executed.
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    print(f"\n✅  Remove BG Tool running at http://0.0.0.0:{port}\n")
    # host='0.0.0.0' -> reachable from outside container (required on Render)
    # threaded=True  -> lets one slow AI request not block every other request
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
