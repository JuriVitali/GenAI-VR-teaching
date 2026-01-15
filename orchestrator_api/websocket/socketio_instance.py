from flask_socketio import SocketIO

socketio = SocketIO(
    cors_allowed_origins="*",
    ping_interval=90,
    ping_timeout=80,
    async_mode="eventlet"
)